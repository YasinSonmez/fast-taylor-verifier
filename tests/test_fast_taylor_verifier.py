"""Tests of fast_taylor_verifier. Needs a CUDA GPU (the first run also builds the kernels).

  compile           cuda.compile agrees with TaylorModel (orders 1-3, Chebyshev on/off) and its bounds hold
  soundness         Taylor-model bounds (orders 1-3, with and without Chebyshev interpolation) contain the values of
                    functions built from every supported operation, at sampled points of each box
  dual              Dual and jacobian bounds contain the Jacobian at sampled points
  network           taylor.network bounds contain a tanh MLP's values and directional derivatives
  constants         the derivative peaks in taylor.py are true upper bounds, and taylor.cuh uses the same numbers
  search            branch_and_bound proves a true property, finds the failing boxes of a false one, and stops early
  batched_code      code written for (B, n) tensors runs on a TaylorModel directly and its bounds hold

usage: python tests/test_fast_taylor_verifier.py [name ...]      (or: pytest tests)
"""
import os
import sys
import time

import torch
import torch.nn as nn

from fast_taylor_verifier import Dual, Settings, TaylorModel, branch_and_bound, cuda, evaluate, jacobian, network, taylor

DEV = "cuda"
SETTINGS = [Settings(order=o, cheb=c) for o in (1, 2, 3) for c in (False, True)]


def _boxes(B, n, seed, scale=1.0):
    """B random boxes in [-2, 2]^n with half-widths between 2^-8 and 1/2 (times scale)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x0 = (4 * torch.rand(B, n, device=DEV, generator=g) - 2) * scale
    h = scale * 2.0 ** -torch.randint(1, 9, (B, n), device=DEV, generator=g).float()
    return x0, h


def _points(x0, h, P, seed):
    """P points per box (B, P, n): the corners first, then uniform samples."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = x0.shape[1]
    d = 2 * torch.rand(len(x0), P, n, device=DEV, generator=g) - 1
    corners = 2 * torch.cartesian_prod(*[torch.tensor([0.0, 1.0], device=DEV)] * n).reshape(-1, n) - 1
    d[:, :len(corners)] = corners
    return x0[:, None] + h[:, None] * d


def _contains(lo, hi, y, what):
    """lo <= y <= hi up to float32 rounding (Taylor models do not model rounding)."""
    slack = 1e-5 * (lo.abs() + hi.abs() + y.abs()) + 1e-6
    bad = (y < lo - slack) | (y > hi + slack)
    assert not bad.any(), f"{what}: {int(bad.sum())} sampled values outside the bounds"
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all(), f"{what}: infinite bounds"


def _f(x):
    """A function of 3 inputs that uses every operation TaylorModel supports through torch."""
    a, b, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([a * b - torch.sin(c) + 0.5,
                      torch.cos(a + b) / (2.0 + b * b),
                      torch.exp(0.3 * c) - torch.tanh(2.0 * a * c),
                      1.0 - a / (3.0 + torch.exp(b)),
                      torch.log(2.5 + b ** 2) - c ** 3 / 8.0], 1)


def _g(x):
    """_f written for one point (x[i] is input i), as evaluate, jacobian and cuda.compile take it."""
    a, b, c = x
    return (a * b - torch.sin(c) + 0.5,
            torch.cos(a + b) / (2.0 + b * b),
            torch.exp(0.3 * c) - torch.tanh(2.0 * a * c),
            1.0 - a / (3.0 + torch.exp(b)),
            torch.log(2.5 + b ** 2) - c ** 3 / 8.0)


# ------------------------------------------------------------------------------------------------------------ tests
def test_soundness():
    x0, h = _boxes(256, 3, seed=1)
    pts = _points(x0, h, 64, seed=2)
    y = _f(pts.reshape(-1, 3).double()).reshape(len(x0), -1, 5)
    for s in SETTINGS:
        lo, hi = _f(TaylorModel.box(x0, h, s)).bounds()
        _contains(lo[:, None].double(), hi[:, None].double(), y, f"f, {s}")
        g_lo, g_hi = evaluate(_g, TaylorModel.box(x0, h, s)).bounds()            # the same operations, one point
        assert torch.equal(g_lo, lo) and torch.equal(g_hi, hi), f"evaluate(_g) differs from _f, {s}"
        sec = TaylorModel.box(x0, h, s)[:, 0:1].sech2()                        # the methods without a torch name
        lg = (TaylorModel.box(x0, h, s)[:, 1:2] * 0.5 + 2.0).log()
        _contains(*[t[:, None].double() for t in sec.bounds()], 1 - torch.tanh(pts[..., 0:1].double()) ** 2,
                  f"sech2, {s}")
        _contains(*[t[:, None].double() for t in lg.bounds()], torch.log(pts[..., 1:2].double() * 0.5 + 2.0),
                  f"log, {s}")


def test_dual():
    x0, h = _boxes(128, 3, seed=3, scale=0.5)
    p = torch.tensor([0.3, -0.7, 1.1], device=DEV)                              # _g on an ordinary point
    assert torch.allclose(torch.stack(_g(p)), _f(p[None])[0]), "_g and _f differ on a point"
    pts = _points(x0, h, 32, seed=4).reshape(-1, 3).double()
    J = torch.func.vmap(torch.func.jacrev(lambda p: _f(p[None])[0]))(pts).reshape(len(x0), -1, 5, 3)
    for s in (Settings(order=2, cheb=True), Settings(order=3)):
        for k in range(3):
            x = TaylorModel.box(x0, h, s)
            e = torch.zeros_like(x0)
            e[:, k] = 1.0
            lo, hi = _f(Dual(x, x._zero(e))).t.bounds()
            _contains(lo[:, None].double(), hi[:, None].double(), J[..., k], f"Jacobian column {k}, {s}")
        jlo, jhi = jacobian(_g, TaylorModel.box(x0, h, s))                          # all columns at once
        _contains(jlo[:, None].double(), jhi[:, None].double(), J, f"jacobian, {s}")


def test_network():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1)).to(DEV)
    with torch.no_grad():                      # larger weights than the default init: strongly nonlinear on the boxes
        for m in mlp:
            if isinstance(m, nn.Linear):
                m.weight *= 3.0
    net = nn.Module()
    net.s = mlp
    x0, h = _boxes(256, 3, seed=5)
    pts = _points(x0, h, 128, seed=6).reshape(-1, 3).double()
    t = torch.tensor([1.0, -2.0, 0.5], device=DEV)
    m64 = mlp.double()
    with torch.enable_grad():
        p = pts.clone().requires_grad_(True)
        v = m64(p)
        g, = torch.autograd.grad(v.sum(), p)
    v, dv = v.detach().reshape(len(x0), -1, 1), (g @ t.double()).reshape(len(x0), -1, 1)
    mlp.float()
    for s in SETTINGS:
        x = TaylorModel.box(x0, h, s)
        val, along = network(net, x)
        d = along(x._zero(t.expand(len(x0), 3)))
        _contains(*[b[:, None].double() for b in val.bounds()], v, f"network value, {s}")
        _contains(*[b[:, None].double() for b in d.bounds()], dv, f"network derivative, {s}")


def test_constants():
    t = torch.linspace(0, 6, 6_000_001, dtype=torch.float64)
    th = torch.tanh(t)
    curves = {"TANH3_PEAKS": (2 * (1 - th ** 2) * (3 * th ** 2 - 1)).abs(),
              "SECH2_3_PEAKS": (8 * (1 - th ** 2) * th * (2 - 3 * th ** 2)).abs(),
              "SECH2_4_PEAKS": (8 * (1 - th ** 2) * (2 - 15 * th ** 2 + 15 * th ** 4)).abs()}
    header = open(os.path.join(os.path.dirname(taylor.__file__), "taylor.cuh")).read()
    for name, y in curves.items():
        peaks = getattr(taylor, name)
        inner = (y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:])
        locs = [0.0] * bool(y[0] >= y[1]) + t[1:-1][inner].tolist()               # local maxima on [0, 6]
        assert len(locs) == len(peaks), f"{name}: {len(locs)} local maxima, {len(peaks)} listed"
        for (p, v), q in zip(peaks, locs):
            true = float(y[(t - q).abs().argmin()])
            assert abs(p - q) < 1e-5, f"{name}: peak at {q:.6f}, listed at {p}"
            assert v >= true and v - true < 1e-3 * true + 1e-4, f"{name}: peak value {true:.6f}, listed {v}"
            for num in (p, v):                                                     # the CUDA kernels use the same
                if num != 0.0:
                    assert f"{num}f" in header or f"{num:g}f" in header, f"{name}: {num} not in taylor.cuh"


def test_compile():
    """cuda.compile(_g) against _f on TaylorModels (the same rules, different rounding), and sampled values."""
    bound = cuda.compile(_g)
    x0, h = _boxes(512, 3, seed=11)
    pts = _points(x0, h, 64, seed=12)
    y = _f(pts.reshape(-1, 3).double()).reshape(len(x0), -1, 5)
    for s in SETTINGS:
        lo, hi = bound(cuda.Box(x0, h, s))
        ref_lo, ref_hi = _f(TaylorModel.box(x0, h, s)).bounds()
        _contains(lo[:, None].double(), hi[:, None].double(), y, f"compiled f, {s}")
        for a, b in ((lo, ref_lo), (hi, ref_hi)):
            err = (a - b).abs() / (a.abs() + b.abs() + 1e-3)
            assert float(err.max()) < 1e-3, f"compiled f, {s}: relative difference {float(err.max()):.2e} from PyTorch"


def test_search():
    """branch_and_bound on f(x) = x0^2 + x1^2 - 0.5 x0 x1 (minimum 0.375 on [0.5, 1]^2, at (0.5, 0.5)), with first-order
    models (they need splits)."""
    def f(x):
        return x[:, 0:1] * x[:, 0:1] + x[:, 1:2] * x[:, 1:2] - 0.5 * x[:, 0:1] * x[:, 1:2]

    def above(t):                                  # open: the boxes where f > t is not proved
        def check(L, U):
            lo, _ = f(TaylorModel.box((L + U) / 2, (U - L) / 2, Settings(order=1))).bounds()   # order 2 is exact
            return lo[:, 0] <= t, None
        return check
    L, U = torch.full((1, 2), 0.5, device=DEV), torch.ones(1, 2, device=DEV)
    r = branch_and_bound(L, U, above(0.37), batch=64, h_min=1e-4)
    assert r["failed"] == 0 and r["queued"] == 0 and r["boxes"] > 1, f"f > 0.37 not proved: {r}"
    r = branch_and_bound(L, U, above(0.38), batch=64, h_min=1e-3)
    assert r["failed"] > 0, f"f > 0.38 is false at (0.5, 0.5), but no box failed: {r}"
    r = branch_and_bound(L, U, above(0.38), batch=4, h_min=1e-6, stop=lambda: True)
    assert r["boxes"] == 0 and r["queued"] == 1, f"stop() was ignored: {r}"


def test_batched_code():
    def f(x):                                  # written for (B, n) tensors, run on a TaylorModel directly
        return torch.cat([x[:, 1:2], -torch.sin(x[:, 0:1]) - 0.1 * x[:, 1:2]], 1)
    centres = torch.zeros(4, 2, device=DEV)
    half_widths = torch.full((4, 2), 0.1, device=DEV)
    x = TaylorModel.box(centres, half_widths, Settings(order=2, cheb=True))
    lo, hi = f(x).bounds()
    pts = _points(centres, half_widths, 64, seed=8)
    _contains(lo[:, None].double(), hi[:, None].double(), f(pts.reshape(-1, 2).double()).reshape(4, -1, 2),
              "batched code")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    names = sys.argv[1:] or [k[5:] for k in globals() if k.startswith("test_")]
    failed = 0
    for name in names:
        t0 = time.time()
        try:
            globals()[f"test_{name}"]()
            print(f"pass  {name}  ({time.time() - t0:.1f} s)", flush=True)
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"{len(names) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
