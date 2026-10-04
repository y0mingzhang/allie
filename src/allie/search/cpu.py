"""Search oracle on the dense port: every node reruns its whole prefix. A CPU reference, not fast."""
import numpy as np
import torch
from .model import DenseBackend, load_checkpoint
from .board import encode, advance_clocks, predicted_seconds, root_other_previous


class CPUOracle:
    def __init__(self, checkpoint=None, *, model=None):
        self.model = model if model is not None else load_checkpoint(checkpoint, "cpu")[0]
        self.model.eval()
        self.reset()

    def reset(self):
        self.new_tokens = 0

    def handles(self, prefixes, features, clock_rule="predicted"):
        return CPUHandles(self, prefixes, features, clock_rule)

    @torch.inference_mode()
    def predict(self, prefix, features):
        ids = torch.tensor(prefix, dtype=torch.long)
        states = torch.from_numpy(encode(ids.numpy()[None])[0])
        logits = self.model(ids, torch.arange(len(prefix)), DenseBackend(),
                            torch.as_tensor(features), states)
        self.new_tokens += len(prefix)
        return logits[-1].float().numpy()


class CPUHandles:
    def __init__(self, base, prefixes, features, clock_rule):
        self.base, self.clock_rule = base, clock_rule
        self.root_logits = np.array([base.predict(p, f) for p, f in zip(prefixes, features)])
        self.queries = 0
        self.per_root_queries = np.zeros(len(prefixes), np.int64)
        self.nodes = {}
        for i, (p, f, z) in enumerate(zip(prefixes, features, self.root_logits)):
            inc = p[2] - 10 if 10 <= p[2] < 191 else -1
            self.nodes[i] = (p, np.asarray(f), root_other_previous(p, f, inc), inc, z, i)

    def __call__(self, handles):
        result = []
        for node, parent, token, length in handles:
            if int(node) in self.nodes:
                raise ValueError("a node must be evaluated once")
            p, f, previous, inc, z, owner = self.nodes[int(parent)]
            elapsed = predicted_seconds(z[None]) if self.clock_rule == "predicted" else np.zeros(1)
            next_features, next_previous = advance_clocks(f[-1:], np.array([previous]),
                np.array([len(p)]), np.array([inc]), elapsed)
            next_features = next_features.astype(np.float32)
            next_previous = next_previous.astype(np.float32)
            pp = p + [int(token)]
            if len(pp) != length:
                raise ValueError("invalid node length")
            ff = np.concatenate((f, next_features))
            zz = self.base.predict(pp, ff)
            self.nodes[int(node)] = (pp, ff, next_previous[0], inc, zz, owner)
            self.queries += 1
            self.per_root_queries[owner] += 1
            result.append(zz)
        return np.array(result)
