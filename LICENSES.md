# Licenses

- This repository is released under the MIT license: [LICENSE](LICENSE).
- [scripts/modded_medium_core.py](scripts/modded_medium_core.py) is derived from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) (commit `ecbb586`) and keeps its MIT notice: [scripts/modded_medium_LICENSE](scripts/modded_medium_LICENSE).
- [search/](search/) derives from the original [Allie](https://github.com/ippolito-cmu/allie) and keeps its MIT notice: [search/ALLIE_LICENSE](search/ALLIE_LICENSE). The chess rules library it builds against is vendored outside the repository under its own license.
- Maia-3 is not included. The benchmark scripts in [bench/](bench/) load the public [Maia-3 checkpoints](https://huggingface.co/collections/MaiaChess/maia3) and run the authors' [inference code](https://github.com/CSSLab/maia-chess) (AGPL-3.0), each under its own license.
