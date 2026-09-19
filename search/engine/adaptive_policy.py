"""Budget and output-policy choices independent of MCTS selection.

These are explicitly named research variants, not changes to the released Allie
baseline. Only model predictions may enter budget allocation. Calibration is fit
on development games and frozen before golden evaluation.
"""
import numpy as np
from scipy.special import logsumexp, softmax


def allocate(signal, mean, minimum=1, maximum=None):
    """Deterministic capped proportional allocation with an exact total budget."""
    signal = np.asarray(signal, np.float64)
    assert signal.ndim == 1 and len(signal) and np.isfinite(signal).all()
    assert (signal >= 0).all() and mean == int(mean)
    maximum = 4 * mean if maximum is None else maximum
    assert 0 <= minimum <= mean <= maximum
    target = int(mean) * len(signal)
    if signal.sum() == 0:
        return np.full(len(signal), int(mean), np.int32)
    # Zeros get a small positive weight so caps cannot make the target infeasible.
    weight = np.maximum(signal, signal.max() * 1e-12)
    lo, hi = 0., maximum / weight.min()
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(mid * weight, minimum, maximum).sum() < target:
            lo = mid
        else:
            hi = mid
    real = np.clip((lo + hi) / 2 * weight, minimum, maximum)
    count = np.floor(real).astype(np.int64)
    extra = target - int(count.sum())
    assert 0 <= extra <= len(count)
    order = np.argsort(-(real - count), kind='stable')
    count[order[:extra]] += 1
    assert count.sum() == target and (count >= minimum).all() and (count <= maximum).all()
    return count.astype(np.int32)


def output(logits, values, legal, alpha=1., beta=1., direction='forward'):
    """Full-support Q-regularized policy; beta is independent of simulation count.

    forward: maximize E_pi Q - KL(pi || calibrated_prior) / beta.
    reverse: maximize E_pi Q - KL(calibrated_prior || pi) / beta.
    beta=0 is the calibrated legal prior. The reverse solution uses FP64 bisection.
    """
    logits, values = np.asarray(logits, np.float64), np.asarray(values, np.float64)
    legal = np.asarray(legal, bool)
    assert logits.shape == values.shape == legal.shape and (legal.sum(1) > 0).all()
    assert beta >= 0 and alpha > 0
    z = np.where(legal, alpha * logits, -np.inf)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    if beta == 0:
        return np.exp(logp)
    if direction == 'forward':
        return softmax(np.where(legal, logp + beta * values, -np.inf), axis=1)
    assert direction == 'reverse'
    prior = np.exp(logp)
    q = np.where(legal, values, -np.inf)
    lam = 1. / beta
    lo = (q + lam * prior).max(1)
    hi = q.max(1) + lam
    for _ in range(80):
        mid = (lo + hi) / 2
        p = lam * prior / (mid[:, None] - q)
        above = p.sum(1) > 1
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    p = lam * prior / (((lo + hi) / 2)[:, None] - q)
    return p / p.sum(1, keepdims=True)
