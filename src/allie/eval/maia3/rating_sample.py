"""The accuracy-by-game-rating sample: every scored blitz move of the golden eval (all 402,108 that
eval.maia3.positions's games.npz lists, not its 80K subsample), as maia3-bench/rating/games.npz.
Only golden-eval games: they passed eval.build's dev/test, leak and BOT exclusions."""

import numpy as np

from allie.eval.maia3.aggregate import DATA


def main():
    with np.load(DATA / "games.npz") as z:
        d = dict(z)
    d["keep"] = np.arange(len(d["sel"]))
    np.savez(DATA / "rating" / "games.npz", **d)
    print(len(d["keep"]), "positions,", len(d["meta"]), "games")


if __name__ == "__main__":
    main()
