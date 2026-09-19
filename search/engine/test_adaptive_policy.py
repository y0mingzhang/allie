"""Analytic-policy and exact-budget invariants for experimental MCTS outputs."""
import numpy as np
from .adaptive_policy import allocate, output


def main():
    rng = np.random.default_rng(18472)
    for n in (1, 17, 512):
        for mean in (16, 64):
            for signal in (np.zeros(n), np.ones(n), rng.lognormal(0, 4, n)):
                ns = allocate(signal, mean)
                assert ns.sum() == n * mean
                assert (ns >= 1).all() and (ns <= mean * 4).all()
                assert np.array_equal(ns, allocate(signal, mean))
    z = rng.normal(size=(61, 41)); q = rng.uniform(-1, 1, z.shape)
    legal = rng.random(z.shape) < .7; legal[:, 0] = True
    for direction in ('forward', 'reverse'):
        for beta in (0., .1, 1., 10.):
            p = output(z, q, legal, .9, beta, direction)
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
            assert (p[legal] > 0).all() and (p[~legal] == 0).all()
            shift = rng.normal(size=(len(z), 1))
            np.testing.assert_allclose(p, output(z, q + shift, legal, .9, beta, direction), atol=1e-12)
            if beta:
                prior = output(z, q, legal, .9, 0)
                # KKT gradient must be constant across each row's legal actions.
                grad = q - np.log(p / np.maximum(prior, 1e-300) + 1e-300) / beta if direction == 'forward' else q + prior / np.maximum(p, 1e-300) / beta
                for g, mask in zip(grad, legal):
                    np.testing.assert_allclose(g[mask], g[mask][0], atol=1e-9)
    print('PASS: exact deterministic capped budgets; full support; value-offset invariance; KL KKT solutions')


if __name__ == '__main__':
    main()
