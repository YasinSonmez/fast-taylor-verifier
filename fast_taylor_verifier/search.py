"""Branch and bound over boxes, for any property that bounds on a box can prove.

branch_and_bound(L, U, check, ...) keeps a queue of boxes [L, U] (rows of (B, n) tensors), takes up to `batch` of them
at a time and asks check(L, U) which boxes are still open:
  check(L, U) -> (open, score)  open: (B,) bool, the boxes that are neither proved nor otherwise settled by check (a
                                counterexample, say, which check records itself); score: per-side split scores of the
                                open boxes (open.sum(), n), or None to split along the widest side.
An open box whose half-width (largest side, in units of `scale`) is at most h_min counts as failed; the others are
halved (see split) and queued. The property holds on [L, U] if the queue empties with no failed box and check found
nothing against it. check may keep state between calls (a level that only decreases, say); `stop` ends the search
early. A CUDA out-of-memory error halves the batch and retries.

Example: prove f(x) = x0^2 + x1^2 - 0.5 x0 x1 > 0.1 on the box [0.5, 1]^2 with Taylor models:

    def check(L, U):
        lo, _ = f(TaylorModel.box((L + U) / 2, (U - L) / 2)).bounds()
        return lo[:, 0] <= 0.1, None
    r = branch_and_bound(torch.full((1, 2), 0.5), torch.ones(1, 2), check, batch=1024, h_min=1e-4)
    proved = r["failed"] == 0 and r["queued"] == 0
"""
import math
import time

import torch


def split(L, U, score=None, scale=1.0, aspect=0.0, tie_break=0.0, h_min=0.0):
    """Halve each box along its side of largest score, among the sides at least aspect x its widest side and wider
    than 2 h_min (side lengths in units of scale); ties go to the longer side (tie_break). Without scores, along the
    widest side. Returns the halves as (L, U) of twice the rows: all first halves, then all second halves."""
    w = (U - L) / scale
    if score is not None:
        keep = (w >= aspect * w.amax(1, keepdim=True)) & (w > 2 * h_min)
        w = torch.where(keep, score + tie_break * w, torch.zeros_like(w))
    j, i = w.argmax(1), torch.arange(len(L))
    mid = (L + U)[i, j] / 2
    L2, U1 = L.clone(), U.clone()
    U1[i, j], L2[i, j] = mid, mid
    return torch.cat([L, L2]), torch.cat([U1, U])


def branch_and_bound(L, U, check, batch, h_min, scale=1.0, aspect=0.0, tie_break=0.0, deadline=math.inf,
                     max_failed=math.inf, stop=None, log=None, log_every=10.0):
    """Search the boxes [L, U] with check (see the module docstring); the queue is a stack (depth first).

    deadline: time.time() at which to give up; max_failed: give up after more failed boxes; stop(): give up when it
    returns True (checked before each batch); log(boxes, queued): called at most every log_every seconds.
    Returns dict(boxes: boxes checked, failed, queued: boxes left unchecked, batch: the batch size in the end)."""
    done, failed, last = 0, 0, time.time()
    stack = [(L, U)]                               # the queue as a stack of chunks

    def take(n):                                   # up to n boxes from the top of the stack
        parts, k = [], 0
        while stack and k < n:
            Lc, Uc = stack.pop()
            if k + len(Lc) > n:
                r = n - k
                stack.append((Lc[r:], Uc[r:]))
                Lc, Uc = Lc[:r], Uc[:r]
            parts.append((Lc, Uc))
            k += len(Lc)
        if len(parts) == 1:
            return parts[0]
        return torch.cat([q[0] for q in parts]), torch.cat([q[1] for q in parts])

    def queued():
        return sum(len(q[0]) for q in stack)

    while stack and time.time() < deadline and not (stop is not None and stop()):
        Lb, Ub = take(batch)
        if not len(Lb):
            continue
        try:
            open_, score = check(Lb, Ub)
        except RuntimeError as err:
            if "out of memory" not in str(err) or batch == 1:
                raise
            torch.cuda.empty_cache()
            batch //= 2
            stack.append((Lb, Ub))
            continue
        done += len(Lb)
        if log is not None and time.time() - last > log_every:
            last = time.time()
            log(done, queued())
        Lf, Uf = Lb[open_], Ub[open_]
        tiny = ((Uf - Lf) / scale).amax(1) / 2 <= h_min
        failed += int(tiny.sum())
        if failed > max_failed:
            break
        if (~tiny).any():
            sc = None if score is None else score[~tiny]
            stack.append(split(Lf[~tiny], Uf[~tiny], sc, scale, aspect, tie_break, h_min))
    return dict(boxes=done, failed=failed, queued=queued(), batch=batch)
