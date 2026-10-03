"""The by-format companion of the blitz benchmark: 5,000 scored moves from each of the golden eval's 12
bullet / rapid / classical cells, sampled and stored exactly as build_positions.py does the blitz ones,
in maia3-bench/formats/ (games.npz, positions.json; then legal.py formats/)."""

import build_positions as bp

bp.BLITZ = tuple(c for c in range(16) if c not in (4, 5, 6, 7))
bp.PER_CELL = 5000
bp.OUT = bp.OUT / "formats"
bp.main()
