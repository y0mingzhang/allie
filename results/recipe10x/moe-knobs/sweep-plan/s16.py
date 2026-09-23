"""S16 (256 experts, top-16, shared 1/4) vs S18 (top-18) vs C (top-12, 1.5x wide), every width a multiple of 16:
per ladder size the widths, params, active-MLP residual vs the dense MLP and FLOP-matched tokens; per-GPU memory
(moe-perf/memmodel.py, node8c-calibrated) at the big-run shapes."""
import sys
from dataclasses import dataclass
sys.path.insert(0, "/home/yimingz3/src/allie/results/recipe10x/moe-perf")
import memmodel as mm

hid = mm.hidden  # modded_arch.swiglu_hidden: round(8 d / 3 / 16) * 16


def dims(d, k, m=16):
    """modded_arch.moe_dims: shared = round(H / 4 / m) m, expert = round((H - shared) / k / m) m."""
    H = hid(d)
    hs = round(H * 0.25 / m) * m
    return H, hs, round((H - hs) / k / m) * m


@dataclass(frozen=True)
class M(mm.Shape):
    def mats(self):
        H, hs, he = dims(self.d, self.k)
        d, c = self.d, self.L - 1
        return {"attn": (self.L, d, d, 4), "mlp": (1, 2 * H, d, 1), "mlp_proj": (1, H, d, 1),
                "moe_up": (c, 2 * he, d, self.E), "moe": (c, he, d, self.E),
                "mlp_shared_up": (c, 2 * hs, d, 1), "mlp_shared": (c, hs, d, 1)}


POS, V = 45.7, 2432
BASE = {"1e17": (12, 512, 5053), "3e17": (16, 768, 5053), "1e18": (16, 768, 16843)}


def per_token(L, d, k=None):
    """modelexp.per_token + modded_arch.extra_flops (board CNN omitted: the same for every arm at a width)."""
    gates = L + 2 * min(5, L // 2)
    f = 24 * L * d**2 + 2 * d * V + 2 * (gates * (d // 64) * 16 + 64) + 4 * d * L * POS
    if k is None:
        return f + 2 * L * (3 * d * hid(d) - 8 * d * d)
    H, hs, he = dims(d, k)
    return f + 2 * (L - 1) * d * 256 + 2 * (L - 1) * (3 * d * (hs + k * he) - 8 * d * d) + 2 * (3 * d * H - 8 * d * d)


G = 1e9
LADDER = [(10, 448), (12, 512), (14, 640), (16, 768), (18, 896), (20, 1024), (22, 1152), (24, 1280)]
if __name__ == "__main__":
    print("| size | dense MLP | S16 shared / expert, residual | S16 active / total | S18 expert, residual | S18 active / total | dense | S16 tokens 1e17 / 3e17 / 1e18 (B) |")
    for L, d in LADDER:
        H, hs, e16 = dims(d, 16)
        _, _, e18 = dims(d, 18)
        s16, s18, dn = M(d, L, 256, 16), M(d, L, 256, 18), mm.Shape(d, L)
        r = lambda k, e: (hs + k * e) / H - 1
        tok = [bs * per_token(bl, bd) / per_token(L, d, 16) * 524288 / G for bl, bd, bs in BASE.values()]
        print(f"| {L}x{d} | {H} | {hs} / {e16}, {r(16, e16):+.1%} | {s16.active/G:.3f} / {s16.total/G:.2f} "
              f"| {e18}, {r(18, e18):+.1%} | {s18.active/G:.3f} / {s18.total/G:.2f} | {dn.total/G:.3f} | "
              + " / ".join(f"{t:.2f}" for t in tok) + " |")
    print("\n| big-run shape | design | expert | active / total (B) | 8 x L40S 32K / 64K ZeRO-2 | L40S 64K ZeRO-3 experts | 8 x H100 64K ZeRO-2 | persistent |")
    for d, L in ((1536, 24), (2048, 24)):
        for name, k in (("S16", 16), ("S18", 18), ("C", 12)):
            s = M(d, L, 256, k)
            cap = mm.budget("8 x L40S") / G
            f = lambda x: f"{x:.1f}" + ("" if x <= cap else " (no)")
            z = [mm.peak(s, 8, T) / G for T in (32768, 65536)]
            print(f"| {d} x {L} | {name} | {dims(d, k)[2]} | {s.active/G:.2f} / {s.total/G:.2f} | {f(z[0])} / {f(z[1])} "
                  f"| {f(mm.peak(s, 8, 65536, 'zero3e') / G)} | {mm.peak(s, 8, 65536) / G:.1f} | {mm.states(s, 8) / G:.1f} |")
    print(f"budgets: 8 x L40S {mm.budget('8 x L40S')/G:.1f} GB, 8 x H100 {mm.budget('8 x H100')/G:.1f} GB")
    print("\n4-GPU peak at 16K micro (sweep lanes):", ", ".join(f"S16 {L}x{d} {mm.peak(M(d, L, 256, 16), 4, 16384)/G:.1f} GB" for L, d in LADDER[-3:]),
          ", dense 24x1280", f"{mm.peak(mm.Shape(1280, 24), 4, 16384)/G:.1f} GB")
