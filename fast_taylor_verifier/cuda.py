"""CUDA kernels for the Taylor models of taylor.py: the same rules (taylor.TaylorModel), without writing every
intermediate model to GPU memory.

  dynamics(sys, x)            the drift a = f(x, 0) and input columns b_j = f(x, e_j) - f(x, 0) on boxes: sys.f is
                              traced once (operator overloading) into straight-line CUDA code over models, one thread
                              per box; any dynamics built from + - * /, sin, cos, tanh, exp and slicing.
  network(net, x, tangents)   a tanh MLP and its derivatives along tangent models (forward mode, taylor.network): one
                              thread block holds a few boxes, one thread one neuron; a layer sums the previous layer's
                              models from shared memory, the unary and product rules run in registers.
  compile(f)                  a function f written for one point (x[i] is input i), built from + - * /, integer powers,
                              sin, cos, tanh, exp and log, traced once into a kernel like dynamics: compile(f)(x) ->
                              bounds (lo, hi) of each output on the boxes x, one thread per box.
  build(...)                  compile your own kernel on these rules: its source includes "taylor.cuh" (the rules) and,
                              to reuse the network kernel, "network.cu" (launch_network).

The boxes are a `Box(c, h, settings)`: centres and half-widths (B, n) and the taylor.Settings. The kernels are compiled
on first use for each state dimension, tangent count and settings, and cached in ~/.cache/torch_extensions. A model of
n inputs is stored as its unique coefficients [c, L (n), Q (n(n+1)/2, i <= j), T (order 3, i <= j <= k), r] (Q and T
symmetric). Rounding is not modelled, as in taylor.py.
"""
import collections
import hashlib
import os
import weakref

import torch
from torch.utils.cpp_extension import load_inline

from .taylor import pointwise

Box = collections.namedtuple("Box", "c h settings")
HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


def num_coeffs(n, order):
    """Packed coefficients of one model: c, L (n), Q (i <= j), T (i <= j <= k, order 3), r."""
    return 2 + n + n * (n + 1) // 2 + (n * (n + 1) * (n + 2) // 6 if order == 3 else 0)


def flags(s, **defines):
    """Compile flags: the settings s, and -D<name>=<value> for each define."""
    return (["-O3", f"-DTM_ORDER={s.order}", f"-DTM_CHEB={int(s.cheb)}", f"-DTM_CHEB_MIN={s.cheb_min_width!r}f"]
            + [f"-D{k}={v}" for k, v in defines.items()])


def build(prefix, signature, source, function, flags):
    """Compile (or load from the cache) a CUDA extension: `source` defines `function`, declared by `signature` (C++).
    The package folder is on the include path (taylor.cuh, network.cu). The extension's name carries a hash of
    everything that goes into it, so a change to any source file or flag builds a new one."""
    tag = hashlib.sha1("\n".join([_read("taylor.cuh"), _read("network.cu"), source] + flags).encode()).hexdigest()[:12]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "{}.{}".format(*torch.cuda.get_device_capability()))
    return load_inline(f"ftv_{prefix}_{tag}", cpp_sources=signature, cuda_sources=source, functions=[function],
                       extra_cuda_cflags=flags, extra_include_paths=[HERE], verbose=False)


# ------------------------------------------------------------------------------------------------------- network
_WEIGHTS = weakref.WeakKeyDictionary()


def mlp_weights(net):
    """The tanh MLP net.s laid out for the network kernel: (W0, b0, WhT, bh, Wo, bo, H, D). Cached per network object
    (for as long as it lives), so pass a new network after changing weights in place."""
    hit = _WEIGHTS.get(net)
    if hit is not None:
        return hit
    lin = [m for m in net.s if isinstance(m, torch.nn.Linear)]
    H = lin[0].out_features
    assert all(m.out_features == H for m in lin[:-1]) and lin[-1].out_features == 1, "equal hidden widths, 1 output"
    if len(lin) > 2:                                       # hidden-to-hidden layers, stacked
        WhT = torch.stack([m.weight.T for m in lin[1:-1]]).contiguous()
        bh = torch.stack([m.bias for m in lin[1:-1]]).contiguous()
    else:                                                  # one hidden layer: unused placeholders
        WhT, bh = lin[0].weight.new_zeros(1, H, H), lin[0].weight.new_zeros(1, H)
    w = (lin[0].weight.contiguous(), lin[0].bias.contiguous(), WhT, bh, lin[-1].weight[0].contiguous(),
         float(lin[-1].bias[0]), H, len(lin) - 1)
    _WEIGHTS[net] = w
    return w


_NET = {}


@torch.no_grad()
def network(net, x, tangents):
    """The tanh MLP net.s and its derivatives along the packed tangent models (B, K, n, NC) on the boxes x, as packed
    models (B, 1 + K, NC) (taylor.network, fused)."""
    n, k = x.c.shape[1], tangents.shape[1]
    key = (x.settings, n, k)
    if key not in _NET:
        signature = ("void net(torch::Tensor x0, torch::Tensor h, torch::Tensor tin, torch::Tensor W0, torch::Tensor b0, "
                     "torch::Tensor WhT, torch::Tensor bh, torch::Tensor Wo, double bo, torch::Tensor out, int64_t H, "
                     "int64_t D);")
        _NET[key] = build(f"net_n{n}k{k}", signature, _read("network.cu"), "net", flags(x.settings, TM_N=n, TM_K=k))
    W0, b0, WhT, bh, Wo, bo, H, D = mlp_weights(net)
    out = torch.empty(len(x.c), k + 1, tangents.shape[-1], device=x.c.device)
    _NET[key].net(x.c.contiguous(), x.h.contiguous(), tangents, W0, b0, WhT, bh, Wo, bo, out, H, D)
    return out


# ---------------------------------------------------------------------------------------------- dynamics, compiled
class _Sym:
    """A traced value of sys.f: per feature either a Taylor-model variable (a C array name) or a constant
    (a C float expression). As in TaylorModel, constants act as scalars and models follow the model rules."""

    def __init__(self, tr, feats):
        self.tr, self.feats = tr, feats                        # feats: [("tm", name) | ("const", expr)]

    def __getitem__(self, idx):                                # x[:, i:j]
        _, j = idx
        assert isinstance(j, slice), "fast_taylor_verifier: index features by slices, x[:, i:i + 1]"
        return _Sym(self.tr, self.feats[j])

    def _bin(self, o, op):
        o = self.tr.lift(o)
        a, b = self.feats, o.feats
        if len(a) == 1 and len(b) > 1:
            a = a * len(b)
        if len(b) == 1 and len(a) > 1:
            b = b * len(a)
        assert len(a) == len(b), "feature count mismatch"
        return _Sym(self.tr, [self.tr.binary(op, x, y) for x, y in zip(a, b)])

    def __add__(self, o): return self._bin(o, "+")
    def __radd__(self, o): return self.tr.lift(o)._bin(self, "+")
    def __sub__(self, o): return self._bin(o, "-")
    def __rsub__(self, o): return self.tr.lift(o)._bin(self, "-")
    def __mul__(self, o): return self._bin(o, "*")
    def __rmul__(self, o): return self.tr.lift(o)._bin(self, "*")
    def __truediv__(self, o): return self._bin(o, "/")
    def __rtruediv__(self, o): return self.tr.lift(o)._bin(self, "/")
    def __neg__(self): return self._map("neg")

    def __pow__(self, k):                                      # integer powers k >= 1, by repeated products
        assert int(k) == k and k >= 1, "fast_taylor_verifier: only integer powers k >= 1"
        out = self
        for _ in range(int(k) - 1):
            out = out * self
        return out

    def _map(self, fn): return _Sym(self.tr, [self.tr.unary(fn, x) for x in self.feats])
    def sin(self): return self._map("sin")
    def cos(self): return self._map("cos")
    def tanh(self): return self._map("tanh")
    def exp(self): return self._map("exp")
    def log(self): return self._map("log")
    def reciprocal(self): return 1.0 / self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.cat:
            parts = args[0]
            tr = next(p.tr for p in parts if isinstance(p, _Sym))
            return _Sym(tr, [f for p in parts for f in tr.lift(p).feats])
        names = {torch.sin: "sin", torch.cos: "cos", torch.tanh: "tanh", torch.exp: "exp", torch.log: "log"}
        if func in names:
            return args[0]._map(names[func])
        name = getattr(func, "__name__", "")
        if name == "__pow__":
            return args[0] ** args[1]
        a, b = args[0], args[1] if len(args) > 1 else None
        ops = {"add": "+", "__add__": "+", "__radd__": "+", "mul": "*", "__mul__": "*", "__rmul__": "*",
               "sub": "-", "__sub__": "-", "div": "/", "__truediv__": "/"}
        if name in ops:
            tr = (a if isinstance(a, _Sym) else b).tr
            return tr.lift(a)._bin(b, ops[name])
        if name == "__rsub__":
            tr = (a if isinstance(a, _Sym) else b).tr
            return tr.lift(b)._bin(a, "-")
        raise NotImplementedError(f"fast_taylor_verifier: dynamics operation {func} is not supported")


class _Tracer:
    """Records sys.f on symbolic inputs as straight-line C code over Taylor models (the rules of taylor.cuh)."""

    def __init__(self, n):
        self.n, self.lines, self.k = n, [], 0

    def tm(self):
        self.k += 1
        name = f"t{self.k}"
        self.lines.append(f"float {name}[NC];")
        return name

    def lift(self, o):
        if isinstance(o, _Sym):
            return o
        if isinstance(o, torch.Tensor):
            assert o.numel() == 1 or o.dim() == 2, "constant tensors: scalars, or (B, k) inputs"
            if o.dim() == 2 and o.numel() != 1:
                raise NotImplementedError("fast_taylor_verifier: per-box tensors in the dynamics (other than u)")
            o = float(o)
        return _Sym(self, [("const", f"{float(o)!r}f")])

    def binary(self, op, x, y):
        (kx, vx), (ky, vy) = x, y
        if kx == "const" and ky == "const":
            return ("const", f"({vx} {op} {vy})")
        L = self.lines
        if op in "+-":
            out = self.tm()
            if kx == "tm" and ky == "tm":                      # TM + TM, TM - TM (= TM + (-TM)): remainders add
                L.append(f"for (int p = 0; p < RR; ++p) {out}[p] = {vx}[p] {op} {vy}[p]; "
                         f"{out}[RR] = {vx}[RR] + {vy}[RR];")
            elif kx == "tm":                                   # TM +- c
                L.append(f"copy<N>({out}, {vx}); {out}[0] {op}= {vy};")
            else:                                              # c + TM, c - TM = (-TM) + c
                L.append(f"copy<N>({out}, {vy});" + (f" neg<N>({out});" if op == "-" else "") + f" {out}[0] += {vx};")
            return ("tm", out)
        if op == "*":
            out = self.tm()
            if kx == "tm" and ky == "tm":
                L.append(f"copy<N>({out}, {vy}); mul<N>({vx}, {out}, h);")
            else:
                t, c = (vx, vy) if kx == "tm" else (vy, vx)
                L.append(f"copy<N>({out}, {t}); scale<N>({out}, {c});")
            return ("tm", out)
        if op == "/":                                          # TM / c = TM * (1/c); a / TM = a * (1/TM)
            if ky == "const":
                return self.binary("*", x, ("const", f"(1.0f / {vy})"))
            inv = self.tm()
            L.append(f"copy<N>({inv}, {vy}); recip_tm<N>({inv}, h);")
            return self.binary("*", ("tm", inv), x)
        raise NotImplementedError(op)

    def unary(self, fn, x):
        k, v = x
        if k == "const":
            c = {"neg": f"(-{v})", "sin": f"sinf({v})", "cos": f"cosf({v})", "tanh": f"tanhf({v})", "exp": f"expf({v})",
                 "log": f"logf({v})"}
            return ("const", c[fn])
        out = self.tm()
        call = {"neg": f"neg<N>({out});", "sin": f"sin_tm<N>({out}, h);", "cos": f"cos_tm<N>({out}, h);",
                "tanh": f"tanh_tm<N>({out}, h);", "exp": f"exp_tm<N>({out}, h);", "log": f"log_tm<N>({out}, h);"}[fn]
        self.lines.append(f"copy<N>({out}, {v}); {call}")
        return ("tm", out)


def _trace(fn, n, m=None):
    """Trace fn on a box of n variables x0[i] + d_i (and, with m, constant inputs u[0..m-1]: fn(x, u)): the C body
    (statements over the arrays x0, h, u) and the output features [("tm", array) | ("const", expr)]."""
    tr = _Tracer(n)
    xs = []
    for i in range(n):
        name = tr.tm()
        tr.lines.append(f"for (int p = 0; p < NC; ++p) {name}[p] = 0.f; {name}[0] = x0[{i}]; {name}[{1 + i}] = 1.f;")
        xs.append(("tm", name))
    args = [_Sym(tr, xs)] + ([_Sym(tr, [("const", f"u[{j}]") for j in range(m)])] if m is not None else [])
    out = tr.lift(fn(*args))
    return "\n        ".join(tr.lines), out.feats


def _dynamics_source(sys):
    """CUDA source of the kernel that writes the packed drift a = f(x, 0) and input columns b_j = f(x, e_j) - f(x, 0)
    on boxes, generated by tracing sys.f."""
    n, m = sys.n, sys.m
    body, feats = _trace(sys.f, n, m)
    assert len(feats) == n
    stores = []
    for i, (k, v) in enumerate(feats):
        if k == "tm":
            stores.append(f"copy<N>(f + {i} * NC, {v});")
        else:
            stores.append(f"for (int p = 0; p < NC; ++p) f[{i} * NC + p] = 0.f; f[{i} * NC] = {v};")
    store = "\n        ".join(stores)
    return '#include "taylor.cuh"\n' + rf"""
template <int N>
__device__ void dyn_eval(const float* x0, const float* h, const float* u, float* f) {{   // f: N models (N x NC)
    constexpr int NC = M<N>::NC, RR = M<N>::R;
        {body}
        {store}
}}

template <int N>
__global__ void dyn_kernel(const float* __restrict__ x0w, const float* __restrict__ hw, float* __restrict__ tin, int B) {{
    // one thread per box; the block's outputs are staged in shared memory and written as one contiguous chunk
    constexpr int NC = M<N>::NC, RR = M<N>::R, S = (1 + {m}) * N * NC;
    extern __shared__ float stage[];
    const int t = threadIdx.x, base = blockIdx.x * blockDim.x, box = base + t;
    const int nb = min((int)blockDim.x, B - base);
    if (t < nb) {{
        float x0[N], h[N], u[{m}] = {{0.f}}, f0[N * NC], f1[N * NC];
        for (int k = 0; k < N; ++k) {{ x0[k] = x0w[box * N + k]; h[k] = hw[box * N + k]; }}
        dyn_eval<N>(x0, h, u, f0);
        float* a = stage + (size_t)t * S;
        for (int q = 0; q < N * NC; ++q) a[q] = f0[q];
        for (int j = 0; j < {m}; ++j) {{                       // b_j = f(x, e_j) - f(x, 0): remainders add
            for (int k = 0; k < {m}; ++k) u[k] = k == j ? 1.f : 0.f;
            dyn_eval<N>(x0, h, u, f1);
            float* b = a + (size_t)(1 + j) * N * NC;
            for (int i = 0; i < N; ++i) {{
                for (int p = 0; p < RR; ++p) b[i * NC + p] = f1[i * NC + p] - f0[i * NC + p];
                b[i * NC + RR] = f1[i * NC + RR] + f0[i * NC + RR];
            }}
        }}
    }}
    __syncthreads();
    float* dst = tin + (size_t)base * S;
    for (int q = t; q < nb * S; q += blockDim.x) dst[q] = stage[q];
}}

void dyn(torch::Tensor x0, torch::Tensor h, torch::Tensor tin) {{
    // boxes per block: 32, or fewer where their staged outputs exceed the device's shared memory per block (6-D, order 3)
    int dev = 0, smax = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    size_t per = (size_t)(1 + {m}) * {n} * M<{n}>::NC * sizeof(float);
    int B = x0.size(0), T = (int)std::max((size_t)1, std::min((size_t)32, (size_t)smax / per));
    size_t shm = (size_t)T * per;
    cudaFuncSetAttribute(dyn_kernel<{n}>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
    dyn_kernel<{n}><<<(B + T - 1) / T, T, shm>>>(x0.data_ptr<float>(), h.data_ptr<float>(), tin.data_ptr<float>(), B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""


_DYN = {}


def dynamics(sys, x):
    """The packed drift and input columns (B, 1 + m, n, NC) on the boxes x, by a kernel compiled from sys.f (sys needs
    .name, .n, .m and .f(x, u); the kernel is cached per name and settings)."""
    key = (sys.name, x.settings)
    if key not in _DYN:
        _DYN[key] = build(f"dyn_{sys.name}", "void dyn(torch::Tensor x0, torch::Tensor h, torch::Tensor tin);",
                           _dynamics_source(sys), "dyn", flags(x.settings))
    columns = torch.empty(len(x.c), 1 + sys.m, sys.n, num_coeffs(sys.n, x.settings.order), device=x.c.device)
    _DYN[key].dyn(x.c.contiguous(), x.h.contiguous(), columns)
    return columns


# ------------------------------------------------------------------------------------------- any function, compiled
def _function_source(fn, n):
    """CUDA source of a kernel that writes the bounds of fn's outputs on boxes (one thread per box), and the number of
    outputs."""
    body, feats = _trace(fn, n)
    k = len(feats)
    stores = []
    for i, (kind, v) in enumerate(feats):
        if kind == "tm":
            stores.append(f"bounds<N>({v}, h, lo[(size_t)box * {k} + {i}], hi[(size_t)box * {k} + {i}]);")
        else:
            stores.append(f"lo[(size_t)box * {k} + {i}] = hi[(size_t)box * {k} + {i}] = {v};")
    store = "\n        ".join(stores)
    return '#include "taylor.cuh"\n' + rf"""
template <int N>
__global__ void fn_kernel(const float* __restrict__ x0w, const float* __restrict__ hw, float* __restrict__ lo,
                          float* __restrict__ hi, int B) {{
    constexpr int NC = M<N>::NC, RR = M<N>::R;
    const int box = blockIdx.x * blockDim.x + threadIdx.x;
    if (box >= B) return;
    float x0[N], h[N];
    for (int k = 0; k < N; ++k) {{ x0[k] = x0w[(size_t)box * N + k]; h[k] = hw[(size_t)box * N + k]; }}
        {body}
        {store}
}}

void fn(torch::Tensor x0, torch::Tensor h, torch::Tensor lo, torch::Tensor hi) {{
    int B = x0.size(0);
    TORCH_CHECK(x0.size(1) == {n}, "fast_taylor_verifier: kernel built for n = {n}, called with n = ", x0.size(1));
    fn_kernel<{n}><<<(B + 127) / 128, 128>>>(x0.data_ptr<float>(), h.data_ptr<float>(), lo.data_ptr<float>(),
                                         hi.data_ptr<float>(), B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
""", k


def compile(fn):
    """Compile fn (written for one point as for taylor.evaluate: x[i] is input i, it returns one value or a tuple;
    + - * /, integer powers, sin, cos, tanh, exp, log) into a CUDA kernel. Returns bound(x): the bounds (lo, hi), each
    (B, k), of fn's k outputs on the boxes x (a Box), the same as evaluate(fn, TaylorModel.box(x.c, x.h,
    x.settings)).bounds() up to rounding. fn is traced once per state dimension and settings, on first use; its Python control flow must not
    depend on the values (it sees symbols, not numbers). Networks: use network (a traced network would unroll every
    weight into the kernel)."""
    kernels = {}

    def bound(x):
        n = x.c.shape[1]
        key = (n, x.settings)
        if key not in kernels:
            source, k = _function_source(lambda s: pointwise(fn, s, n), n)
            name = "".join(ch if ch.isalnum() else "_" for ch in getattr(fn, "__name__", "fn"))[:32]
            kernels[key] = build(f"fn_{name}_n{n}", "void fn(torch::Tensor x0, torch::Tensor h, torch::Tensor lo, "
                                 "torch::Tensor hi);", source, "fn", flags(x.settings)), k
        ext, k = kernels[key]
        lo = torch.empty(len(x.c), k, device=x.c.device)
        hi = torch.empty_like(lo)
        ext.fn(x.c.float().contiguous(), x.h.float().contiguous(), lo, hi)
        return lo, hi
    return bound
