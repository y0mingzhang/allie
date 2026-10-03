# Licenses

- This repository is released under the MIT license: [LICENSE](LICENSE).
- [src/allie/model/nanogpt.py](src/allie/model/nanogpt.py) and the modules built around it derive from [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) (commit `ecbb586`) and keep its MIT notice: [src/allie/model/NOTICE](src/allie/model/NOTICE).
- [src/allie/search/](src/allie/search/) derives from the original [Allie](https://github.com/ippolito-cmu/allie) and keeps its MIT notice: [src/allie/search/ALLIE_LICENSE](src/allie/search/ALLIE_LICENSE). The chess rules library it builds against is vendored outside the repository under its own license.
- Maia-3 is not included. The benchmark scorer ([src/allie/eval/maia3/](src/allie/eval/maia3/)) loads the public [Maia-3 checkpoints](https://huggingface.co/collections/MaiaChess/maia3) and runs the authors' [inference code](https://github.com/CSSLab/maia-chess) (AGPL-3.0), each under its own license.
