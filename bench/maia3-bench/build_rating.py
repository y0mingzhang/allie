"""The accuracy-by-game-rating sample: every scored blitz move of the golden eval (all 402,108 that
build_positions.py's games.npz lists, not its 80K subsample), as maia3-bench/rating/games.npz.
Only golden-eval games: they passed strateval.py's dev/test, leak and BOT exclusions."""

import numpy as np

from aggregate import DATA

with np.load(DATA / "games.npz") as z:
    d = dict(z)
d["keep"] = np.arange(len(d["sel"]))
np.savez(DATA / "rating" / "games.npz", **d)
print(len(d["keep"]), "positions,", len(d["meta"]), "games")
