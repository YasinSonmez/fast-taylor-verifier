// The tanh MLP kernel (see cuda.network): a network and its derivatives along K tangent models, on boxes. Compiled
// once per state dimension TM_N, number of tangents TM_K and settings (TM_ORDER, TM_CHEB, ...: the -D flags of cuda.py).
#pragma once
#include "taylor.cuh"

// grid: boxes / BPB blocks; block (H, BPB). Shared: per box the previous layer's value and K tangent models, H x NC each,
// in rows of SC >= NC floats (a multiple of 4: the sums read them as float4, four coefficients per shared load).
template <int NC> __device__ __forceinline__ void acc_row(float* z, const float* a, float w, float aw) {
    constexpr int RR = NC - 1, SC = (NC + 3) / 4 * 4;    // z += w a (the remainder |w| r)
    #pragma unroll
    for (int q = 0; q < SC / 4; ++q) {
        float4 v = reinterpret_cast<const float4*>(a)[q];
        float e[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int p = 4 * q + t;
            if (p < RR) z[p] += w * e[t];
            else if (p == RR) z[RR] += aw * e[t];
        }
    }
}

template <int N, int K>
__global__ void net_kernel(const float* __restrict__ x0, const float* __restrict__ hw, const float* __restrict__ tin,
                           const float* __restrict__ W0, const float* __restrict__ b0,
                           const float* __restrict__ WhT, const float* __restrict__ bh,
                           const float* __restrict__ Wo, float bo, float* __restrict__ out, int B, int H, int D) {
    constexpr int NC = M<N>::NC, RR = M<N>::R, SC = (NC + 3) / 4 * 4;
    extern __shared__ float4 sh4[];
    float* sh = reinterpret_cast<float*>(sh4);
    const int i = threadIdx.x, y = threadIdx.y;
    const int box = blockIdx.x * blockDim.y + y;
    const bool live = box < B;
    const int bx = live ? box : B - 1;
    float* act = sh + (size_t)y * (1 + K) * H * SC;          // [H][SC], then K tangent buffers [H][SC]
    float h[N], z[NC], u[K][NC];
    #pragma unroll
    for (int k = 0; k < N; ++k) h[k] = hw[bx * N + k];

    for (int layer = 0; layer < D; ++layer) {
        #pragma unroll
        for (int p = 0; p < NC; ++p) {
            z[p] = 0.f;
            #pragma unroll
            for (int k = 0; k < K; ++k) u[k][p] = 0.f;
        }
        if (layer == 0) {                                     // the input is the box: x = x0 + d
            z[0] = b0[i];
            #pragma unroll
            for (int k = 0; k < N; ++k) { float w = W0[i * N + k]; z[0] += w * x0[bx * N + k]; z[1 + k] = w; }
            #pragma unroll
            for (int k = 0; k < K; ++k)                       // tangents: W0 t (t an N-vector of models)
                #pragma unroll
                for (int j = 0; j < N; ++j) {
                    float w = W0[i * N + j];
                    const float* t = tin + (((size_t)bx * K + k) * N + j) * NC;
                    #pragma unroll
                    for (int p = 0; p < RR; ++p) u[k][p] += w * t[p];
                    u[k][RR] += fabsf(w) * t[RR];
                }
        } else {                                              // hidden layer: sums over the previous layer's neurons
            const float* Wl = WhT + (size_t)(layer - 1) * H * H;
            z[0] = bh[(layer - 1) * H + i];
            #pragma unroll 4
            for (int j = 0; j < H; ++j) {
                float w = Wl[j * H + i], aw = fabsf(w);
                acc_row<NC>(z, act + j * SC, w, aw);
                #pragma unroll
                for (int k = 0; k < K; ++k) acc_row<NC>(u[k], act + ((size_t)(1 + k) * H + j) * SC, w, aw);
            }
        }
        __syncthreads();                                      // everyone has read the previous layer
        float s[NC];
        #pragma unroll
        for (int p = 0; p < NC; ++p) s[p] = z[p];
        tanh_tm<N>(z, h);                                     // activation
        sech2_tm<N>(s, h);                                    // slope
        #pragma unroll
        for (int p = 0; p < NC; ++p) act[i * SC + p] = z[p];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            mul<N>(s, u[k], h);                               // tangent: sech^2(z) (W t)
            #pragma unroll
            for (int p = 0; p < NC; ++p) act[((size_t)(1 + k) * H + i) * SC + p] = u[k][p];
        }
        __syncthreads();
    }
    for (int q = i; q < (1 + K) * NC; q += H) {               // output layer: s, and the K derivatives (no bias)
        int which = q / NC, p = q % NC;
        const float* buf = act + (size_t)which * H * SC;
        float acc = (which == 0 && p == 0) ? bo : 0.f;
        for (int j = 0; j < H; ++j) acc += (p == RR ? fabsf(Wo[j]) : Wo[j]) * buf[j * SC + p];
        if (live) out[((size_t)box * (1 + K) + which) * NC + p] = acc;
    }
}

// Launches net_kernel: out (B, 1 + K, NC) = [s, grad s . t_1, ..., grad s . t_K] for the tangents tin (B, K, N, NC).
template <int N, int K>
void launch_network(torch::Tensor x0, torch::Tensor h, torch::Tensor tin, torch::Tensor W0, torch::Tensor b0,
                    torch::Tensor WhT, torch::Tensor bh, torch::Tensor Wo, double bo, torch::Tensor out, int H, int D) {
    constexpr int SC = (M<N>::NC + 3) / 4 * 4;
    size_t per_box = (size_t)(1 + K) * H * SC * sizeof(float);          // shared memory of one box's models
    int dev = 0, smax = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    TORCH_CHECK(per_box <= (size_t)smax, "fast_taylor_verifier: one box needs ", per_box, " B of shared memory, more than ",
                "the device's ", smax, " B per block (network too wide for this state dimension and order)");
    int B = x0.size(0), bpb = (H >= 128 || per_box > 24576) ? 1 : 128 / H; // large models: one box per block
    bpb = std::max(1, std::min(bpb, (int)((size_t)smax / per_box)));       // and never more than fit in a block
    size_t shm = (size_t)bpb * per_box;
    auto kern = net_kernel<N, K>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    dim3 block(H, bpb), grid((B + bpb - 1) / bpb);
    kern<<<grid, block, shm>>>(x0.data_ptr<float>(), h.data_ptr<float>(), tin.data_ptr<float>(), W0.data_ptr<float>(),
                               b0.data_ptr<float>(), WhT.data_ptr<float>(), bh.data_ptr<float>(), Wo.data_ptr<float>(),
                               (float)bo, out.data_ptr<float>(), B, H, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();                           // a failed launch raises instead of leaving `out` unset
}

void net(torch::Tensor x0, torch::Tensor h, torch::Tensor tin, torch::Tensor W0, torch::Tensor b0, torch::Tensor WhT,
         torch::Tensor bh, torch::Tensor Wo, double bo, torch::Tensor out, int64_t H, int64_t D) {
    int N = x0.size(1), K = tin.size(1);
    TORCH_CHECK(N == TM_N && K == TM_K, "fast_taylor_verifier: kernels built for n = ", TM_N, ", K = ", TM_K,
                ", called with n = ", N, ", K = ", K);
    launch_network<TM_N, TM_K>(x0, h, tin, W0, b0, WhT, bh, Wo, bo, out, H, D);
}
