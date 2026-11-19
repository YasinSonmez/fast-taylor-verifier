// Taylor-model rules on packed coefficients (the CUDA mirror of taylor.py's TaylorModel), shared by certificate.cu
// and the dynamics kernels that cuda.py generates from a system's f.
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>              // C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef TM_ORDER
#define TM_ORDER 2                               // 1: first-order models for comparison; 3: third-order models
#endif

template <int N> struct M {                      // coefficient layout of one model
    static constexpr int NS = N * (N + 1) / 2;
    static constexpr int NT = TM_ORDER == 3 ? N * (N + 1) * (N + 2) / 6 : 0;
    static constexpr int NC = 1 + N + NS + NT + 1;   // c, L, Q (i <= j), T (i <= j <= k, order 3), r
    static constexpr int R = NC - 1;
};

template <int N> __device__ __forceinline__ int qi(int i, int j) {   // index of Q_ij (i <= j) in the packed model
    return 1 + N + i * N - i * (i - 1) / 2 + (j - i);
}

template <int N> __device__ __forceinline__ int ti(int i, int j, int k) {   // index of T_ijk (i <= j <= k)
    int p = 0;
    for (int a = 0; a < i; ++a) p += (N - a) * (N - a + 1) / 2;
    for (int b = i; b < j; ++b) p += N - b;
    return 1 + N + M<N>::NS + p + (k - j);
}

__device__ __forceinline__ float mult3(int i, int j, int k) {   // distinct orderings of (i, j, k), i <= j <= k
    return (i == j && j == k) ? 1.f : ((i == j || j == k) ? 3.f : 6.f);
}

template <int N> struct Ranges { float R, S, qlo, qhi, off, U; };

__device__ __forceinline__ float top(float l, float q, float hb) {   // max of l d + q d^2 over |d| <= hb, exactly
    bool interior = (q < 0.f) && (fabsf(l) < -2.f * q * hb);
    return interior ? l * l / fmaxf(-4.f * q, 1e-30f) : fabsf(l) * hb + q * hb * hb;
}

template <int N> __device__ __forceinline__ Ranges<N> ranges(const float* v, const float* h) {
    Ranges<N> o{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float l = v[1 + i], q = v[qi<N>(i, i)];
        o.R += fabsf(l) * h[i];
        o.S += fabsf(q) * h[i] * h[i];
        o.qhi += top(l, q, h[i]);
        o.qlo -= top(-l, -q, h[i]);
        #pragma unroll
        for (int j = i + 1; j < N; ++j) o.off += 2.f * fabsf(v[qi<N>(i, j)]) * h[i] * h[j];
    }
    o.S += o.off;
#if TM_ORDER == 3                                // max |T[d, d, d]| <= sum |T_ijk| h_i h_j h_k over all orderings
    #pragma unroll
    for (int i = 0; i < N; ++i)
        #pragma unroll
        for (int j = i; j < N; ++j)
            #pragma unroll
            for (int k = j; k < N; ++k) o.U += mult3(i, j, k) * fabsf(v[ti<N>(i, j, k)]) * h[i] * h[j] * h[k];
#endif
    return o;
}

template <int N> __device__ __forceinline__ void fold(float* v, const float* h) {   // order 1: Q into the remainder
#if TM_ORDER == 1
    Ranges<N> g = ranges<N>(v, h);
    v[M<N>::R] += g.S;
    #pragma unroll
    for (int p = 1 + N; p < M<N>::R; ++p) v[p] = 0.f;
#endif
}

template <int N> __device__ __forceinline__ void unary(float* v, const float* h, float fc, float a1, float a2, float a3,
                                                     float mr) {
    // f(c + t), a_k = f^(k)(c) / k!. Order <= 2: L' = a1 L, Q' = a1 Q + a2 L L^T,
    // r' = |a1| r + |a2| (2 R (S + r) + (S + r)^2) + mr W^3 / 6, mr bounding |f^(3)| on the range.
    // Order 3 (TaylorModel._unary3): also T' = a1 T + 2 a2 sym(L Q) + a3 L L L; with D = S + U + r, W = R + D,
    // r' = |a1| r + |a2| (D^2 + 2 R (U + r)) + |a3| D (W^2 + W R + R^2) + mr W^4 / 24, mr bounding |f^(4)|.
    Ranges<N> g = ranges<N>(v, h);
#if TM_ORDER == 3
    {
        float r = v[M<N>::R], D = g.S + g.U + r, W = g.R + D;
        #pragma unroll
        for (int i = 0; i < N; ++i)              // T first: it reads the old L and Q
            #pragma unroll
            for (int j = i; j < N; ++j)
                #pragma unroll
                for (int k = j; k < N; ++k) {
                    float li = v[1 + i], lj = v[1 + j], lk = v[1 + k];
                    float lq = (li * v[qi<N>(j, k)] + lj * v[qi<N>(i, k)] + lk * v[qi<N>(i, j)]) / 3.f;
                    v[ti<N>(i, j, k)] = a1 * v[ti<N>(i, j, k)] + 2.f * a2 * lq + a3 * li * lj * lk;
                }
        #pragma unroll
        for (int i = 0; i < N; ++i)
            #pragma unroll
            for (int j = i; j < N; ++j) v[qi<N>(i, j)] = a1 * v[qi<N>(i, j)] + a2 * v[1 + i] * v[1 + j];
        #pragma unroll
        for (int i = 0; i < N; ++i) v[1 + i] *= a1;
        v[M<N>::R] = fabsf(a1) * r + fabsf(a2) * (D * D + 2.f * g.R * (g.U + r))
                     + fabsf(a3) * D * (W * W + W * g.R + g.R * g.R) + mr * W * W * W * W / 24.f;
        v[0] = fc;
        return;
    }
#endif
    float r = v[M<N>::R], sr = g.S + r, W = g.R + sr;
    #pragma unroll
    for (int i = 0; i < N; ++i)
        #pragma unroll
        for (int j = i; j < N; ++j) v[qi<N>(i, j)] = a1 * v[qi<N>(i, j)] + a2 * v[1 + i] * v[1 + j];
    #pragma unroll
    for (int i = 0; i < N; ++i) v[1 + i] *= a1;
    v[M<N>::R] = fabsf(a1) * r + fabsf(a2) * (2.f * g.R * sr + sr * sr) + mr * W * W * W / 6.f;
    v[0] = fc;
    fold<N>(v, h);
}

#ifndef TM_CHEB
#define TM_CHEB 0                                // 1: unary functions also try Chebyshev interpolation (Settings.cheb)
#endif

// f(u) with the quadratic interpolating f at 3 Chebyshev nodes of u's range [lo, hi] when its remainder is smaller:
// |f - p| <= mr (w/2)^3 / 24 there, and the constant stays f(c) (the difference |p(c) - f(c)| goes into the remainder),
// so the centre values are exact. Otherwise (and for narrow ranges, where divided differences lose precision): unary.
template <int N, class F>
__device__ __forceinline__ void unaryf(float* v, const float* h, F f, float fc, float a1, float a2, float a3, float mr) {
#if TM_CHEB && TM_ORDER <= 2
    Ranges<N> g = ranges<N>(v, h);
    float lo = v[0] + g.qlo - g.off - v[M<N>::R], hi = v[0] + g.qhi + g.off + v[M<N>::R];
    float w = hi - lo, mid = 0.5f * (hi + lo);
    if (isfinite(w) && w > TM_CHEB_MIN * (1.f + fabsf(mid))) {
        float r = v[M<N>::R], sr = g.S + r, W = g.R + sr, tr = 2.f * g.R * sr + sr * sr;
        float rT = fabsf(a1) * r + fabsf(a2) * tr + mr * W * W * W / 6.f;
        float x0 = mid + 0.4330127f * w, x1 = mid, x2 = mid - 0.4330127f * w;   // mid + (w/2) cos((2k+1) pi / 6)
        float f0 = f(x0), f1 = f(x1), f2 = f(x2);
        float F1 = (f1 - f0) / (x1 - x0), F2 = ((f2 - f1) / (x2 - x1) - F1) / (x2 - x0);
        float u0 = v[0] - x0, u1 = v[0] - x1, hw = 0.5f * w;
        float p0 = f0 + F1 * u0 + F2 * u0 * u1, p1 = F1 + F2 * (u0 + u1), p2 = F2;
        float rC = fabsf(p1) * r + fabsf(p2) * tr + mr * hw * hw * hw / 24.f + fabsf(p0 - fc);
        if (isfinite(rC) && rC < rT) {
            #pragma unroll
            for (int i = 0; i < N; ++i)
                #pragma unroll
                for (int j = i; j < N; ++j) v[qi<N>(i, j)] = p1 * v[qi<N>(i, j)] + p2 * v[1 + i] * v[1 + j];
            #pragma unroll
            for (int i = 0; i < N; ++i) v[1 + i] *= p1;
            v[M<N>::R] = rC;
            v[0] = fc;
            fold<N>(v, h);
            return;
        }
    }
#endif
#if TM_CHEB && TM_ORDER == 3                     // the cubic interpolating f at 4 Chebyshev nodes (TaylorModel._unary3):
    {                                            // |f - p| <= mr (w/2)^4 / 192, mr bounding |f^(4)|; constant f(c)
        Ranges<N> g = ranges<N>(v, h);
        float r = v[M<N>::R], lo = v[0] + g.qlo - g.off - g.U - r, hi = v[0] + g.qhi + g.off + g.U + r;
        float w = hi - lo, mid = 0.5f * (hi + lo);
        if (isfinite(w) && w > TM_CHEB_MIN * (1.f + fabsf(mid))) {
            float D = g.S + g.U + r, W = g.R + D;
            float tr2 = D * D + 2.f * g.R * (g.U + r), tr3 = D * (W * W + W * g.R + g.R * g.R);
            float rT = fabsf(a1) * r + fabsf(a2) * tr2 + fabsf(a3) * tr3 + mr * W * W * W * W / 24.f;
            float x0 = mid + 0.4619398f * w, x1 = mid + 0.1913417f * w, x2 = mid - 0.1913417f * w, x3 = mid - 0.4619398f * w;
            float f0 = f(x0), f1 = f(x1), f2 = f(x2), f3 = f(x3);
            float F1 = (f1 - f0) / (x1 - x0), G1 = (f2 - f1) / (x2 - x1), H1 = (f3 - f2) / (x3 - x2);
            float F2 = (G1 - F1) / (x2 - x0), G2 = (H1 - G1) / (x3 - x1), F3 = (G2 - F2) / (x3 - x0);
            float u0 = v[0] - x0, u1 = v[0] - x1, u2 = v[0] - x2, hw = 0.5f * w;
            float p0 = f0 + F1 * u0 + F2 * u0 * u1 + F3 * u0 * u1 * u2;
            float p1 = F1 + F2 * (u0 + u1) + F3 * (u0 * u1 + u0 * u2 + u1 * u2), p2 = F2 + F3 * (u0 + u1 + u2), p3 = F3;
            float rC = fabsf(p1) * r + fabsf(p2) * tr2 + fabsf(p3) * tr3 + mr * hw * hw * hw * hw / 192.f + fabsf(p0 - fc);
            if (isfinite(rC) && rC < rT) {
                #pragma unroll
                for (int i = 0; i < N; ++i)      // T first: it reads the old L and Q
                    #pragma unroll
                    for (int j = i; j < N; ++j)
                        #pragma unroll
                        for (int k = j; k < N; ++k) {
                            float li = v[1 + i], lj = v[1 + j], lk = v[1 + k];
                            float lq = (li * v[qi<N>(j, k)] + lj * v[qi<N>(i, k)] + lk * v[qi<N>(i, j)]) / 3.f;
                            v[ti<N>(i, j, k)] = p1 * v[ti<N>(i, j, k)] + 2.f * p2 * lq + p3 * li * lj * lk;
                        }
                #pragma unroll
                for (int i = 0; i < N; ++i)
                    #pragma unroll
                    for (int j = i; j < N; ++j) v[qi<N>(i, j)] = p1 * v[qi<N>(i, j)] + p2 * v[1 + i] * v[1 + j];
                #pragma unroll
                for (int i = 0; i < N; ++i) v[1 + i] *= p1;
                v[M<N>::R] = rC;
                v[0] = fc;
                return;
            }
        }
    }
#endif
    unary<N>(v, h, fc, a1, a2, a3, mr);
}

__device__ __forceinline__ float pmax(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : fmaxf(a, b); }
__device__ __forceinline__ float pmin(float a, float b) { return (isnan(a) || isnan(b)) ? NAN : fminf(a, b); }
__device__ __forceinline__ float nan_to(float a, float v) { return isnan(a) ? v : a; }

// |f^(k)| of tanh and sech^2 for the remainder bounds; the peak locations and values below are TANH3_PEAKS,
// SECH2_3_PEAKS and SECH2_4_PEAKS of taylor.py (tests.py checks them).
__device__ __forceinline__ float tanh3(float t) { float th = tanhf(t); return fabsf(2.f * (1.f - th * th) * (3.f * th * th - 1.f)); }
__device__ __forceinline__ float sech23(float t) { float th = tanhf(t); return fabsf(8.f * (1.f - th * th) * th * (2.f - 3.f * th * th)); }
__device__ __forceinline__ float sech24(float t) { float th = tanhf(t), u = th * th; return fabsf(8.f * (1.f - u) * (2.f - 15.f * u + 15.f * u * u)); }
__device__ __forceinline__ bool hits(float lo, float hi, float p) { return (lo <= p && hi >= p) || (lo <= -p && hi >= -p); }

template <int N> __device__ __forceinline__ void bounds(const float* v, const float* h, float& lo, float& hi) {
    Ranges<N> g = ranges<N>(v, h);
    lo = v[0] + g.qlo - g.off - g.U - v[M<N>::R];
    hi = v[0] + g.qhi + g.off + g.U + v[M<N>::R];
}

template <int N> __device__ __forceinline__ void tanh_tm(float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi);
    float m = pmax(tanh3(lo), tanh3(hi));
    if (hits(lo, hi, 0.f)) m = pmax(m, 2.0f);
    if (hits(lo, hi, 1.146216f)) m = pmax(m, 0.66667f);
    float th = tanhf(v[0]), s = 1.f - th * th;
#if TM_ORDER == 3                                // |tanh^(4)| = |(sech^2)^(3)|
    m = pmax(sech23(lo), sech23(hi));
    if (hits(lo, hi, 0.421316f)) m = pmax(m, 4.08589f);
    if (hits(lo, hi, 1.572159f)) m = pmax(m, 0.61009f);
#endif
    unaryf<N>(v, h, [](float x) { return tanhf(x); }, th, s, 0.5f * (-2.f * th * s), 2.f * s * (3.f * th * th - 1.f) / 6.f, m);
}

template <int N> __device__ __forceinline__ void sech2_tm(float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi);
    float m = pmax(sech23(lo), sech23(hi));
    if (hits(lo, hi, 0.421316f)) m = pmax(m, 4.08589f);
    if (hits(lo, hi, 1.572159f)) m = pmax(m, 0.61009f);
    float th = tanhf(v[0]), s = 1.f - th * th;
#if TM_ORDER == 3                                // max |(sech^2)^(4)| on the range (taylor._sech2_4, SECH2_4_PEAKS in taylor.py)
    m = pmax(sech24(lo), sech24(hi));
    if (hits(lo, hi, 0.f)) m = pmax(m, 16.0016f);
    if (hits(lo, hi, 0.75668f)) m = pmax(m, 7.6875f);
    if (hits(lo, hi, 1.96805f)) m = pmax(m, 0.5757f);
#endif
    unaryf<N>(v, h, [](float x) { float t = tanhf(x); return 1.f - t * t; }, s, -2.f * th * s, 0.5f * s * (6.f * th * th - 2.f),
              8.f * th * s * (2.f - 3.f * th * th) / 6.f, m);
}

template <int N> __device__ __forceinline__ void mul(const float* a, float* b, const float* h) {
    // b <- a b: keep up to L_a L_b, bound the rest (TaylorModel.__mul__); order 3 keeps degree 3
    Ranges<N> ga = ranges<N>(a, h), gb = ranges<N>(b, h);
    float ca = a[0], cb = b[0], ra = a[M<N>::R], rb = b[M<N>::R];
#if TM_ORDER == 3
    #pragma unroll
    for (int i = 0; i < N; ++i)                  // T first: it reads the old L and Q of b
        #pragma unroll
        for (int j = i; j < N; ++j)
            #pragma unroll
            for (int k = j; k < N; ++k) {
                float lq = (a[1 + i] * b[qi<N>(j, k)] + a[1 + j] * b[qi<N>(i, k)] + a[1 + k] * b[qi<N>(i, j)]
                            + b[1 + i] * a[qi<N>(j, k)] + b[1 + j] * a[qi<N>(i, k)] + b[1 + k] * a[qi<N>(i, j)]) / 3.f;
                b[ti<N>(i, j, k)] = ca * b[ti<N>(i, j, k)] + cb * a[ti<N>(i, j, k)] + lq;
            }
#endif
    #pragma unroll
    for (int i = 0; i < N; ++i)
        #pragma unroll
        for (int j = i; j < N; ++j)
            b[qi<N>(i, j)] = ca * b[qi<N>(i, j)] + cb * a[qi<N>(i, j)] + 0.5f * (a[1 + i] * b[1 + j] + b[1 + i] * a[1 + j]);
    #pragma unroll
    for (int i = 0; i < N; ++i) b[1 + i] = ca * b[1 + i] + cb * a[1 + i];
#if TM_ORDER == 3
    b[M<N>::R] = fabsf(ca) * rb + fabsf(cb) * ra + ga.R * gb.U + ga.U * gb.R + ga.S * gb.S + ga.S * gb.U + ga.U * gb.S
                 + ga.U * gb.U + ra * (gb.R + gb.S + gb.U + rb) + rb * (ga.R + ga.S + ga.U);
#else
    b[M<N>::R] = fabsf(ca) * rb + fabsf(cb) * ra + ga.R * gb.S + ga.S * gb.R + ga.S * gb.S + (ga.R + ga.S) * rb
                 + ra * (gb.R + gb.S + rb);
#endif
    b[0] = ca * cb;
    fold<N>(b, h);
}

template <int N> __device__ __forceinline__ void exp_tm(float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi);
    float e = expf(v[0]);
    unaryf<N>(v, h, [](float x) { return expf(x); }, e, e, 0.5f * e, e / 6.f, expf(hi));       // |exp^(k)| <= e^hi
}

template <int N> __device__ __forceinline__ void log_tm(float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi);
    float c = v[0], l3 = fmaxf(lo, 1e-30f), m = lo > 0.f ? 2.f / (l3 * l3 * l3) : INFINITY;
#if TM_ORDER == 3
    m = lo > 0.f ? 6.f / (l3 * l3 * l3 * l3) : INFINITY;
#endif
    if (isnan(lo)) m = INFINITY;
    unaryf<N>(v, h, [](float x) { return logf(x); }, logf(c), 1.f / c, 0.5f * (-1.f / (c * c)), (2.f / (c * c * c)) / 6.f, m);
}

template <int N> __device__ __forceinline__ void recip_tm(float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi);
    float mn = fminf(fabsf(lo), fabsf(hi)), c = v[0];
    float m = (lo <= 0.f && hi >= 0.f) ? INFINITY : 6.f / (mn * mn * mn * mn);
#if TM_ORDER == 3
    m = (lo <= 0.f && hi >= 0.f) ? INFINITY : 24.f / (mn * mn * mn * mn * mn);
#endif
    if (isnan(lo) || isnan(hi)) m = NAN;
    unaryf<N>(v, h, [](float x) { return 1.f / x; }, 1.f / c, -1.f / (c * c), 0.5f * (2.f / (c * c * c)), -1.f / (c * c * c * c), m);
}

template <int N> __device__ __forceinline__ void axpy(float* y, float a, const float* x) {   // y += a x (models)
    #pragma unroll
    for (int p = 0; p < M<N>::R; ++p) y[p] += a * x[p];
    y[M<N>::R] += fabsf(a) * x[M<N>::R];
}

template <int N> __device__ __forceinline__ void add(float* y, const float* x) {           // y += x
    #pragma unroll
    for (int p = 0; p < M<N>::NC; ++p) y[p] += x[p];
}

template <int N> __device__ __forceinline__ float upper(const float* v, const float* h) {
    float lo, hi; bounds<N>(v, h, lo, hi); return hi;
}

template <int N> __device__ __forceinline__ void sin_tm(float* v, const float* h) {
    float c = v[0], s = sinf(c);
    unaryf<N>(v, h, [](float x) { return sinf(x); }, s, cosf(c), 0.5f * (-s), -cosf(c) / 6.f, 1.f);
}

template <int N> __device__ __forceinline__ void cos_tm(float* v, const float* h) {
    float c = v[0], co = cosf(c);
    unaryf<N>(v, h, [](float x) { return cosf(x); }, co, -sinf(c), 0.5f * (-co), sinf(c) / 6.f, 1.f);
}

template <int N> __device__ __forceinline__ void scale(float* v, float a) {             // v <- a v
    #pragma unroll
    for (int p = 0; p < M<N>::R; ++p) v[p] *= a;
    v[M<N>::R] *= fabsf(a);
}

template <int N> __device__ __forceinline__ void neg(float* v) {                         // v <- -v
    #pragma unroll
    for (int p = 0; p < M<N>::R; ++p) v[p] = -v[p];
}

template <int N> __device__ __forceinline__ void copy(float* y, const float* x) {
    #pragma unroll
    for (int p = 0; p < M<N>::NC; ++p) y[p] = x[p];
}
