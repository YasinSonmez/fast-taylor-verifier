"""Taylor models: bounds of smooth functions on a batch of boxes, by operator overloading.

On a box with centre x0 and half-widths h, a quantity is carried as
    u(x0 + d)  in  c + L d + d^T Q d + T[d, d, d] + [-r, r]   for all |d_i| <= h_i
with T only at order 3. Order 2 drops T; order 1 also folds Q into r after every nonlinear operation. Linear operations
are exact; a product keeps the terms up to the order and bounds the rest; a smooth unary f is expanded at c with a
Lagrange remainder from a bound on its next derivative over the argument's range (or, if smaller, replaced by the
polynomial interpolating f at Chebyshev nodes of the range). `TaylorModel` overloads +, -, *, /, slicing and torch.sin,
cos, tanh, exp and cat, so code written for tensors (such as a system's dynamics f) runs on it unchanged. Rounding is
not modelled; a verifier's comparisons should carry a relative margin.

    x = TaylorModel.box(centres, half_widths, Settings(order=2))     # the identity on each box, (B, n)
    lo, hi = f(x).bounds()                                           # f written for (B, n) tensors
    lo, hi = evaluate(g, x).bounds()                                 # g written for one point: g(x) uses x[i]

`Dual` (a model plus a model of its derivative along one direction) bounds Jacobians the same way, and `network`
evaluates a tanh MLP and its directional derivatives on models. cuda.py implements the same rules in CUDA kernels.
"""
import dataclasses
import math

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class Settings:
    """How models are built. Stored on every model (models computed from a box inherit its settings) and compiled
    into the CUDA kernels (cuda.py). The defaults are the values used in the paper."""
    order: int = 2                 # 1, 2 or 3
    cheb: bool = False             # unary functions also try the polynomial interpolating f at Chebyshev nodes,
    cheb_min_width: float = 1e-3   # on argument ranges wider than this x (1 + |midpoint|) (below, divided differences
                                   # lose precision)


# ------------------------------------------------------------------------------------------ math constants
# Facts about tanh, not settings: changing them would make the bounds unsound. tests.py checks each one.
# Peaks of the even functions |f^(k)| used as remainder bounds (t = tanh, s = sech^2): (location, value rounded up).
#   |tanh'''| = |2 s (3 t^2 - 1)|,   |(sech^2)'''| = |8 s t (2 - 3 t^2)| = |tanh''''|,
#   |(sech^2)''''| = |8 s (2 - 15 t^2 + 15 t^4)|.
TANH3_PEAKS = ((0.0, 2.0), (1.146216, 0.66667))
SECH2_3_PEAKS = ((0.421316, 4.08589), (1.572159, 0.61009))
SECH2_4_PEAKS = ((0.0, 16.0016), (0.75668, 7.6875), (1.96805, 0.5757))


def _tanh3(t):
    th = torch.tanh(t)
    return (2 * (1 - th ** 2) * (3 * th ** 2 - 1)).abs()


def _sech2_3(t):
    th = torch.tanh(t)
    return (8 * (1 - th ** 2) * th * (2 - 3 * th ** 2)).abs()


def _sech2_4(t):
    th = torch.tanh(t)
    return (8 * (1 - th ** 2) * (2 - 15 * th ** 2 + 15 * th ** 4)).abs()


def _range_max(g, peaks, lo, hi):
    """The max of an even |f^(k)| over [lo, hi]: its values at the ends and at the peaks +-p inside."""
    m = torch.maximum(g(lo), g(hi))
    for p, v in peaks:
        inside = ((lo <= p) & (hi >= p)) | ((lo <= -p) & (hi >= -p))
        m = torch.where(inside, torch.maximum(m, torch.full_like(m, v)), m)
    return m


# ------------------------------------------------------------------------------------------ helpers
def _poly_ranges(L, Q, h):
    """max |L d|, max |d^T Q d|, the polynomial's min and max without cross terms, and the cross terms' bound."""
    hh = h[:, :, None] * h[:, None, :]
    R = (L.abs() * h[:, None, :]).sum(-1)
    S = (Q.abs() * hh[:, None]).sum((-1, -2))
    q = torch.diagonal(Q, dim1=-2, dim2=-1)
    off = S - (q.abs() * (h ** 2)[:, None]).sum(-1)            # cross terms Q_ij d_i d_j, i != j

    def top(a, q):                                             # max of a d + q d^2 over |d| <= h, exactly (per axis)
        hb = h[:, None, :].expand_as(a)
        interior = (q < 0) & (a.abs() < -2 * q * hb)           # the vertex -a / 2q lies inside
        return torch.where(interior, a * a / (-4 * q).clamp_min(1e-30), a.abs() * hb + q * hb * hb)
    return R, S, -top(-L, -q).sum(-1), top(L, q).sum(-1), off


def _expand2(L, Q, r, R, S, a1, a2, m3):
    """Order <= 2, f(c + t) with t = l + q + e: L' = a1 L, Q' = a1 Q + a2 L L^T, and the remainder (|t| <= R + S + r;
    a1 = f'(c), a2 = f''(c) / 2, m3 bounds |f'''| on the range)."""
    W = R + S + r
    Q = a1[..., None, None] * Q + a2[..., None, None] * (L[..., :, None] * L[..., None, :])
    r = a1.abs() * r + a2.abs() * (2 * R * (S + r) + (S + r) ** 2) + m3 * W ** 3 / 6
    return a1[..., None] * L, Q, r


def _sym3(A):
    """Symmetric part of a (..., n, n, n) tensor (the same cubic form)."""
    p = lambda *o: A.permute(*range(A.dim() - 3), *o)                       # noqa: E731
    return (A + p(-3, -1, -2) + p(-2, -3, -1) + p(-2, -1, -3) + p(-1, -3, -2) + p(-1, -2, -3)) / 6


def _outer_lq(L, Q):                   # L_i Q_jk, symmetrized
    return _sym3(L[..., :, None, None] * Q[..., None, :, :])


def _torch_function(cls, func, args, kwargs):
    """torch.cat, sin, cos, tanh, exp, log and (tensor op model) for the model classes (TaylorModel, Dual). Operators are
    matched by name, which works across PyTorch versions (cbf.py runs in another environment)."""
    if func is torch.cat:
        parts, dim = args[0], kwargs.get("dim", args[1] if len(args) > 1 else 0)
        assert dim == 1
        ref = next(p for p in parts if isinstance(p, cls))
        return ref._cat([p if isinstance(p, cls) else ref._zero(p) for p in parts])
    names = {torch.sin: "sin", torch.cos: "cos", torch.tanh: "tanh", torch.exp: "exp", torch.log: "log"}
    if func in names:
        return getattr(args[0], names[func])()
    name = getattr(func, "__name__", "")
    if name == "__pow__":
        return args[0] ** args[1]
    a, b = args[0], args[1] if len(args) > 1 else None
    if name in ("__add__", "add", "__radd__"):
        return b + a if isinstance(b, cls) else a + b
    if name in ("__mul__", "mul", "__rmul__"):
        return b * a if isinstance(b, cls) else a * b
    if name in ("__sub__", "sub"):
        return (-b) + a if isinstance(b, cls) else a - b
    if name == "__rsub__":
        return a.__rsub__(b) if isinstance(a, cls) else (-a) + b
    if name in ("__truediv__", "div"):
        return b.__rtruediv__(a) if isinstance(b, cls) else a / b
    raise NotImplementedError(f"{cls.__name__}: {func}")


# ------------------------------------------------------------------------------------------ Taylor models
class TaylorModel:
    """A batch of Taylor models of k features on B boxes of dimension n: c (B, k), L (B, k, n), Q (B, k, n, n)
    symmetric, T (B, k, n, n, n) symmetric or None below order 3, r (B, k); h (B, n) the boxes' half-widths."""

    def __init__(self, c, L, Q, T, r, h, settings):
        self.c, self.L, self.Q, self.T, self.r, self.h, self.settings = c, L, Q, T, r, h, settings

    @staticmethod
    def box(x0, h, settings=Settings()):
        """The identity on boxes with centres x0 and half-widths h, both (B, n)."""
        B, n = x0.shape
        L = torch.eye(n, device=x0.device, dtype=x0.dtype).expand(B, n, n).clone()
        T = x0.new_zeros(B, n, n, n, n) if settings.order == 3 else None
        return TaylorModel(x0, L, x0.new_zeros(B, n, n, n), T, torch.zeros_like(x0), h, settings)

    def _new(self, c, L, Q, T, r):             # a model on the same boxes, with the same settings
        return TaylorModel(c, L, Q, T, r, self.h, self.settings)

    def _zero(self, c):                        # a constant on the same boxes
        n = self.h.shape[1]
        T = c.new_zeros(*c.shape, n, n, n) if self.T is not None else None
        return self._new(c, c.new_zeros(*c.shape, n), c.new_zeros(*c.shape, n, n), T, torch.zeros_like(c))

    def _cat(self, parts):
        cat = lambda k: torch.cat([getattr(p, k) for p in parts], 1)       # noqa: E731
        return self._new(cat("c"), cat("L"), cat("Q"), cat("T") if self.T is not None else None, cat("r"))

    def _fold(self):
        """Order 1: the quadratic part goes into the remainder."""
        if self.settings.order == 1:
            R, S, _ = self._ranges()
            return self._new(self.c, self.L, torch.zeros_like(self.Q), None, self.r + S)
        return self

    def _ranges(self):
        """max |L d|, max |d^T Q d| and max |T[d, d, d]| (0 below order 3) over the box; computed once per model."""
        if not hasattr(self, "_R"):
            self._R, self._S, self._qlo, self._qhi, self._off = _poly_ranges(self.L, self.Q, self.h)
            self._U = 0.0
            if self.T is not None:
                hhh = self.h[:, :, None, None] * self.h[:, None, :, None] * self.h[:, None, None, :]
                self._U = (self.T.abs() * hhh[:, None]).sum((-1, -2, -3))
        return self._R, self._S, self._U

    def width(self):
        """max |u - c| over the box."""
        R, S, U = self._ranges()
        return R + S + U + self.r

    def bounds(self):
        """Lower and upper bounds (B, k) over each box: per axis l_i d_i + Q_ii d_i^2 exactly, cross terms by
        |Q_ij| h_i h_j, the cubic part by sum |T_ijk| h_i h_j h_k."""
        _, _, U = self._ranges()
        return self.c + self._qlo - self._off - U - self.r, self.c + self._qhi + self._off + U + self.r

    def scores(self):
        """Each side's share of the model's variation over the box (first feature): |L_j| h_j + sum_k |Q_jk| h_j h_k
        (+ sum_kl |T_jkl| h_j h_k h_l)."""
        h = self.h
        s = self.L[:, 0].abs() * h + (self.Q[:, 0].abs() * h[:, None, :]).sum(-1) * h
        if self.T is not None:
            s = s + (self.T[:, 0].abs() * h[:, None, None, :] * h[:, None, :, None]).sum((-1, -2)) * h
        return s

    # ------------------------------------------------------------------ arithmetic
    def _const(self, o):
        return torch.as_tensor(o, device=self.c.device, dtype=self.c.dtype).expand_as(self.c)

    def __add__(self, o):
        if isinstance(o, TaylorModel):
            T = self.T + o.T if self.T is not None else None
            return self._new(self.c + o.c, self.L + o.L, self.Q + o.Q, T, self.r + o.r)
        return self._new(self.c + o, self.L, self.Q, self.T, self.r)

    __radd__ = __add__

    def __neg__(self):
        return self._new(-self.c, -self.L, -self.Q, -self.T if self.T is not None else None, self.r)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, TaylorModel):     # a constant or a per-box tensor
            o = self._const(o)
            T = self.T * o[..., None, None, None] if self.T is not None else None
            return self._new(self.c * o, self.L * o[..., None], self.Q * o[..., None, None], T, self.r * o.abs())
        R1, S1, U1 = self._ranges()
        R2, S2, U2 = o._ranges()
        c1, c2 = self.c, o.c
        outer = self.L[..., :, None] * o.L[..., None, :]
        Q = c1[..., None, None] * o.Q + c2[..., None, None] * self.Q + 0.5 * (outer + outer.transpose(-1, -2))
        L = c1[..., None] * o.L + c2[..., None] * self.L
        if self.T is None:                     # keep up to l1 l2, bound the rest
            r1, r2 = self.r, o.r
            r = c1.abs() * r2 + c2.abs() * r1 + R1 * S2 + S1 * R2 + S1 * S2 + (R1 + S1) * r2 + r1 * (R2 + S2 + r2)
            return self._new(c1 * c2, L, Q, None, r)._fold()
        T = (c1[..., None, None, None] * o.T + c2[..., None, None, None] * self.T
             + _outer_lq(self.L, o.Q) + _outer_lq(o.L, self.Q))
        r = (c1.abs() * o.r + c2.abs() * self.r + R1 * U2 + U1 * R2 + S1 * S2 + S1 * U2 + U1 * S2 + U1 * U2
             + self.r * (R2 + S2 + U2 + o.r) + o.r * (R1 + S1 + U1))
        return self._new(c1 * c2, L, Q, T, r)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * o.reciprocal() if isinstance(o, TaylorModel) else self * (1.0 / o)

    def __rtruediv__(self, o):
        return self.reciprocal() * o

    def __pow__(self, k):                      # integer powers k >= 1, by repeated products
        assert int(k) == k and k >= 1, "only integer powers k >= 1"
        out = self
        for _ in range(int(k) - 1):
            out = out * self
        return out

    def __getitem__(self, idx):                # slicing along the feature dimension, x[:, i:j]
        _, j = idx
        return self._new(self.c[:, j], self.L[:, j], self.Q[:, j], self.T[:, j] if self.T is not None else None,
                         self.r[:, j])

    def sum(self, dim=1, keepdim=False):
        T = self.T.sum(dim, keepdim) if self.T is not None else None
        return self._new(self.c.sum(dim, keepdim), self.L.sum(dim, keepdim), self.Q.sum(dim, keepdim), T,
                         self.r.sum(dim, keepdim))

    def linear(self, W, b=None):              # x -> x W^T + b on the feature dimension
        c = self.c @ W.T + (0 if b is None else b)
        T = torch.einsum("mk,bkijl->bmijl", W, self.T) if self.T is not None else None
        return self._new(c, torch.einsum("mk,bkn->bmn", W, self.L), torch.einsum("mk,bkij->bmij", W, self.Q), T,
                         self.r @ W.abs().T)

    # ------------------------------------------------------------------ smooth unary functions
    def unary(self, f, d1, d2, m3, m4):
        """f of the model. d1, d2: f' and f'' (orders 1 and 2); m3(lo, hi), m4(lo, hi): bounds of |f'''| and |f''''|
        on the argument's range [lo, hi] (the Lagrange remainders of orders 2 and 3)."""
        if self.T is not None:
            return self._unary3(f, m4)
        return self._unary2(f, d1, d2, m3)

    def _unary2(self, f, d1, d2, m3):
        """f(u) = f(c) + f'(c) t + f''(c) t^2 / 2 + R, t = u - c, |R| <= m3(lo, hi) |t|^3 / 6; or the quadratic
        interpolating f at 3 Chebyshev nodes, |f - p| <= m3 (w/2)^3 / 24."""
        lo, hi = self.bounds()
        R, S, _ = self._ranges()
        M3 = m3(lo, hi)
        L, Q, r = _expand2(self.L, self.Q, self.r, R, S, d1(self.c), 0.5 * d2(self.c), M3)
        c = f(self.c)
        if self.settings.cheb:
            w, mid = hi - lo, (hi + lo) / 2
            x0, x1, x2 = mid + 0.4330127 * w, mid, mid - 0.4330127 * w          # mid + (w/2) cos((2k+1) pi / 6)
            f0, f1, f2 = f(x0), f(x1), f(x2)
            F1 = (f1 - f0) / (x1 - x0)
            F2 = ((f2 - f1) / (x2 - x1) - F1) / (x2 - x0)
            u0, u1 = self.c - x0, self.c - x1                                   # p in t = u - c (the model's centre)
            p0, p1, p2 = f0 + F1 * u0 + F2 * u0 * u1, F1 + F2 * (u0 + u1), F2
            Lc, Qc, rc = _expand2(self.L, self.Q, self.r, R, S, p1, p2, torch.zeros_like(M3))
            rc = rc + M3 * (w / 2) ** 3 / 24 + (p0 - c).abs()      # constant f(c): the centre values stay exact
            ok = (w > self.settings.cheb_min_width * (1 + mid.abs())) & torch.isfinite(rc) & (rc < r)
            L, Q = torch.where(ok[..., None], Lc, L), torch.where(ok[..., None, None], Qc, Q)
            r = torch.where(ok, rc, r)
        return self._new(c, L, Q, None, r)._fold()

    def _unary3(self, f, m4):
        """f(c + t) = f(c) + f' t + f''/2 t^2 + f'''/6 t^3 + R, |R| <= m4(lo, hi) |t|^4 / 24 (f', f'', f''' at c by
        autograd); or the cubic interpolating f at 4 Chebyshev nodes, |f - p| <= m4 (w/2)^4 / 192."""
        lo, hi = self.bounds()
        R, S, U = self._ranges()
        with torch.enable_grad():
            x = self.c.detach().requires_grad_(True)
            y = f(x)
            d1, = torch.autograd.grad(y.sum(), x, create_graph=True)
            d2, = torch.autograd.grad(d1.sum(), x, create_graph=True)
            d3, = torch.autograd.grad(d2.sum(), x)
        a1, a2, a3 = d1.detach(), d2.detach() / 2, d3 / 6
        D = S + U + self.r                                         # |t - l|
        W = R + D                                                  # |t|
        L = a1[..., None] * self.L
        Q = a1[..., None, None] * self.Q + a2[..., None, None] * (self.L[..., :, None] * self.L[..., None, :])
        LLL = self.L[..., :, None, None] * self.L[..., None, :, None] * self.L[..., None, None, :]
        T = (a1[..., None, None, None] * self.T + 2 * a2[..., None, None, None] * _outer_lq(self.L, self.Q)
             + a3[..., None, None, None] * LLL)
        M4 = m4(lo, hi)
        tr2, tr3 = D ** 2 + 2 * R * (U + self.r), D * (W ** 2 + W * R + R ** 2)
        r = a1.abs() * self.r + a2.abs() * tr2 + a3.abs() * tr3 + M4 * W ** 4 / 24
        if self.settings.cheb:
            w, mid = hi - lo, (hi + lo) / 2
            # the nodes mid + (w/2) cos((2k+1) pi / 8), k = 0 .. 3
            xs = [mid + k * w for k in (0.4619398, 0.1913417, -0.1913417, -0.4619398)]
            fs = [f(x) for x in xs]
            F1 = (fs[1] - fs[0]) / (xs[1] - xs[0])
            G1 = (fs[2] - fs[1]) / (xs[2] - xs[1])
            H1 = (fs[3] - fs[2]) / (xs[3] - xs[2])
            F2, G2 = (G1 - F1) / (xs[2] - xs[0]), (H1 - G1) / (xs[3] - xs[1])
            F3 = (G2 - F2) / (xs[3] - xs[0])
            u0, u1, u2 = self.c - xs[0], self.c - xs[1], self.c - xs[2]
            p0 = fs[0] + F1 * u0 + F2 * u0 * u1 + F3 * u0 * u1 * u2
            p1 = F1 + F2 * (u0 + u1) + F3 * (u0 * u1 + u0 * u2 + u1 * u2)
            p2, p3 = F2 + F3 * (u0 + u1 + u2), F3
            yc = y.detach()
            rc = p1.abs() * self.r + p2.abs() * tr2 + p3.abs() * tr3 + M4 * (w / 2) ** 4 / 192 + (p0 - yc).abs()
            ok = (w > self.settings.cheb_min_width * (1 + mid.abs())) & torch.isfinite(rc) & (rc < r)
            Lc = p1[..., None] * self.L
            Qc = p1[..., None, None] * self.Q + p2[..., None, None] * (self.L[..., :, None] * self.L[..., None, :])
            Tc = (p1[..., None, None, None] * self.T + 2 * p2[..., None, None, None] * _outer_lq(self.L, self.Q)
                  + p3[..., None, None, None] * LLL)
            L = torch.where(ok[..., None], Lc, L)
            Q = torch.where(ok[..., None, None], Qc, Q)
            T = torch.where(ok[..., None, None, None], Tc, T)
            r = torch.where(ok, rc, r)
        return self._new(y.detach(), L, Q, T, r)

    def tanh(self):
        th = torch.tanh
        return self.unary(th, lambda t: 1 - th(t) ** 2, lambda t: -2 * th(t) * (1 - th(t) ** 2),
                          lambda lo, hi: _range_max(_tanh3, TANH3_PEAKS, lo, hi),
                          lambda lo, hi: _range_max(_sech2_3, SECH2_3_PEAKS, lo, hi))

    def sech2(self):                           # 1 - tanh^2 = tanh'
        th = torch.tanh
        s2 = lambda t: 1 - th(t) ** 2                                       # noqa: E731
        return self.unary(s2, lambda t: -2 * th(t) * s2(t), lambda t: s2(t) * (6 * th(t) ** 2 - 2),
                          lambda lo, hi: _range_max(_sech2_3, SECH2_3_PEAKS, lo, hi),
                          lambda lo, hi: _range_max(_sech2_4, SECH2_4_PEAKS, lo, hi))

    def sin(self):
        one = lambda lo, hi: torch.ones_like(lo)                            # noqa: E731
        return self.unary(torch.sin, torch.cos, lambda t: -torch.sin(t), one, one)

    def cos(self):
        one = lambda lo, hi: torch.ones_like(lo)                            # noqa: E731
        return self.unary(torch.cos, lambda t: -torch.sin(t), lambda t: -torch.cos(t), one, one)

    def exp(self):
        top = lambda lo, hi: torch.exp(hi)                                  # noqa: E731
        return self.unary(torch.exp, torch.exp, torch.exp, top, top)

    def reciprocal(self):                      # an infinite remainder where the range touches 0
        def bound(k):                          # max |(1/t)^(k)| = k! / min |t|^(k+1)
            def m(lo, hi):
                v = math.factorial(k) / torch.minimum(lo.abs(), hi.abs()) ** (k + 1)
                return torch.where((lo <= 0) & (hi >= 0), torch.full_like(v, float("inf")), v)
            return m
        return self.unary(lambda t: 1 / t, lambda t: -1 / t ** 2, lambda t: 2 / t ** 3, bound(3), bound(4))

    def log(self):                             # an infinite remainder where the range is not positive
        def bound(k):                          # max |log^(k)| = (k-1)! / min t^k
            def m(lo, hi):
                return torch.where(lo > 0, math.factorial(k - 1) / lo.clamp_min(1e-30) ** k,
                                   torch.full_like(lo, float("inf")))
            return m
        f = torch.log if self.T is None else (lambda t: torch.log(t.clamp_min(1e-30)))   # order 3 differentiates f
        return self.unary(f, lambda t: 1 / t, lambda t: -1 / t ** 2, bound(3), bound(4))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return _torch_function(cls, func, args, kwargs or {})


def select(mask, a, b):
    """Per box: model a where mask (B,), else b (same shapes)."""
    def pick(va, vb):
        return None if va is None else torch.where(mask.reshape(-1, *[1] * (va.dim() - 1)), va, vb)
    return a._new(*[pick(getattr(a, k), getattr(b, k)) for k in "cLQTr"])


# ------------------------------------------------------------------------------------- forward mode on models
class Dual:
    """v + eps t with v, t Taylor models: run a function on Dual(x, e_k) to bound its Jacobian column k on boxes."""

    def __init__(self, v, t):
        self.v, self.t = v, t

    c = property(lambda self: self.v.c)                    # code that reads the batch size from .c

    def _zero(self, c):
        return Dual(self.v._zero(c), self.v._zero(torch.zeros_like(c)))

    def _cat(self, parts):
        return Dual(torch.cat([p.v for p in parts], 1), torch.cat([p.t for p in parts], 1))

    def __add__(self, o):
        return Dual(self.v + o.v, self.t + o.t) if isinstance(o, Dual) else Dual(self.v + o, self.t)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.t)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v * o.v, self.v * o.t + self.t * o.v)
        return Dual(self.v * o, self.t * o)

    __rmul__ = __mul__

    def reciprocal(self):
        r = self.v.reciprocal()
        return Dual(r, -(r * r) * self.t)

    def __truediv__(self, o):
        return self * o.reciprocal() if isinstance(o, Dual) else self * (1.0 / o)

    def __rtruediv__(self, o):
        return self.reciprocal() * o

    def __getitem__(self, idx):
        return Dual(self.v[idx], self.t[idx])

    def sum(self, dim=1, keepdim=False):
        return Dual(self.v.sum(dim, keepdim), self.t.sum(dim, keepdim))

    def linear(self, W, b=None):
        return Dual(self.v.linear(W, b), self.t.linear(W))

    def tanh(self):
        return Dual(self.v.tanh(), self.v.sech2() * self.t)

    def sech2(self):
        s = self.v.sech2()
        return Dual(s, (s * self.v.tanh()) * (-2.0) * self.t)

    def sin(self):
        return Dual(self.v.sin(), self.v.cos() * self.t)

    def cos(self):
        return Dual(self.v.cos(), -(self.v.sin() * self.t))

    def exp(self):
        e = self.v.exp()
        return Dual(e, e * self.t)

    def log(self):
        return Dual(self.v.log(), self.v.reciprocal() * self.t)

    __pow__ = TaylorModel.__pow__              # integer powers by repeated products (works for any class with *)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return _torch_function(cls, func, args, kwargs or {})


# ------------------------------------------------------------------------------ functions written for one point
class _Point:
    """One point of the B boxes in x, for a function written for a single point: p[i] is input i of every box, a
    (B, 1) column of x (a model, a Dual, or a traced value in cuda.py)."""

    def __init__(self, x, n):
        self.x, self.n = x, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise TypeError("fast_taylor_verifier: f is written for one point; index its input with an integer, x[i]")
        if not -self.n <= i < self.n:
            raise IndexError(f"input {i} of a point with {self.n} inputs")
        i %= self.n
        return self.x[:, i:i + 1]


def pointwise(f, x, n):
    """f, written for one point (x[i] is input i; it returns one value or a tuple of values), on the B boxes of x
    with n inputs: the (B, k) result."""
    out = f(_Point(x, n))
    return torch.cat(list(out), 1) if isinstance(out, (tuple, list)) else out


def evaluate(f, x):
    """The Taylor model of f on the boxes x (a TaylorModel.box): a (B, k) model, with .bounds(), .c and .scores().
    f is written for one point: x[i] is input i, and it returns one value or a tuple of values."""
    return pointwise(f, x, x.c.shape[1])


def jacobian(f, x):
    """Bounds of the Jacobian of f (written for one point, as for evaluate) on the boxes x (a TaylorModel.box, n
    inputs): lo, hi of shape (B, k, n), one forward-mode pass (Dual) per column."""
    n = x.c.shape[1]
    lo, hi = [], []
    for k in range(n):
        e = torch.zeros_like(x.c)
        e[:, k] = 1.0
        col_lo, col_hi = pointwise(f, Dual(x, x._zero(e)), n).t.bounds()
        lo.append(col_lo)
        hi.append(col_hi)
    return torch.stack(lo, 2), torch.stack(hi, 2)


# ------------------------------------------------------------------------------------------ tanh networks
def network(net, x):
    """A tanh MLP net.s (Linear and Tanh layers, one output) on the model x: s(x), and a function giving the
    directional derivative grad s . t along a model t (forward mode: t <- sech^2(z) (W t) through the layers)."""
    assert all(isinstance(m, (nn.Linear, nn.Tanh)) for m in net.s), "the derivative formula assumes a tanh MLP"
    layers = [m for m in net.s if isinstance(m, nn.Linear)]
    zs, act = [], x
    for lin in layers[:-1]:
        z = act.linear(lin.weight, lin.bias)
        zs.append(z)
        act = z.tanh()
    s = act.linear(layers[-1].weight, layers[-1].bias)            # (B, 1)
    slopes = [z.sech2() for z in zs]

    def along(t):
        for lin, d in zip(layers[:-1], slopes):
            t = d * t.linear(lin.weight)
        return t.linear(layers[-1].weight)
    return s, along
