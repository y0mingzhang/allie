import json
from pathlib import Path
import unittest
import numpy as np
from allie.search.board import encode, advance_boards, advance_clocks


class ContextTests(unittest.TestCase):
    def test_incremental_boards(self):
        rows = json.loads(Path(__file__).with_name("migration.json").read_text())["positions"]
        for r in rows:
            p = r["prefix"]
            full = encode(np.array(p, np.int64)[None])[0]
            np.testing.assert_array_equal(advance_boards(full[10:-1], p[11:]), full[11:])

    def test_own_previous_clock_and_missingness(self):
        parent = np.array([[100., 120., 5.]])
        child, previous = advance_clocks(parent, np.array([7.]), np.array([31]), np.array([2]), np.array([4.]))
        np.testing.assert_array_equal(child, [[120, 98, 7]])
        grand, _ = advance_clocks(child, previous, np.array([32]), np.array([2]), np.array([6.]))
        np.testing.assert_array_equal(grand, [[98, 116, 4]])
        unknown, _ = advance_clocks(np.full((1, 3), -1), np.array([-1]), np.array([31]), np.array([2]), np.array([4.]))
        np.testing.assert_array_equal(unknown, [[-1, -1, -1]])
        first, _ = advance_clocks(parent, np.array([-1]), np.array([11]), np.array([2]), np.array([4.]))
        np.testing.assert_array_equal(first, [[120, 100, -1]])

    def test_cpu_model_backend(self):
        import torch
        from allie.search.cpu import CPUOracle
        from allie.search.model import ChessLM
        torch.set_num_threads(1)
        torch.manual_seed(425)
        cfg = dict(width=32, layers=8, head_dim=8, vocab_size=2432, scalars_size=32,
                   rotary_length=1025, split_embed=True, ws_short=11, ws_long=23,
                   attn_scale=.125, feats=3, mlp_hidden=64, norm_eps=torch.finfo(torch.float32).eps)
        model = ChessLM(cfg).float()
        with torch.no_grad():
            for p in model.parameters():
                p.uniform_(-.02, .02)
            model.cos.fill_(1)
            model.sin.zero_()
        row = json.loads(Path(__file__).with_name("migration.json").read_text())["positions"][0]
        prefix = row["prefix"][:15]
        feats = np.full((len(prefix), 3), -1, np.float32)
        oracle = CPUOracle(model=model)
        bridge = oracle.handles([prefix], [feats], "zero")
        from allie.search.native import from_prefix
        token = from_prefix(prefix).legal()[0]
        got = bridge(np.array([[1, 0, token, len(prefix)+1]], np.int32))
        direct = oracle.predict(prefix+[token], np.concatenate((feats, feats[-1:])))
        np.testing.assert_array_equal(got[0], direct)
        self.assertEqual(bridge.queries, 1)
        self.assertEqual(bridge.per_root_queries.tolist(), [1])
        self.assertEqual(got.shape, (1, 2432))
        self.assertTrue(np.isfinite(got).all())


if __name__ == "__main__":
    unittest.main()
