"""The Hugging Face release: weights, config, the inference code and a model card in one folder.

    python -m allie.lichess.hub build EXPORT_DIR OUT_DIR   # weights hard-linked, not copied
    python -m allie.lichess.hub upload OUT_DIR             # private unless --public

The folder runs with transformers alone (AutoModel, trust_remote_code): it ships this package's
api.py, fast.py, model.py and tokens.py unchanged, plus hf/'s configuration, modeling wrapper and card.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from .api import REPO

GITHUB = "https://github.com/y0mingzhang/allie"
HERE = Path(__file__).parent
CODE = ("api.py", "fast.py", "model.py", "tokens.py")
WRAPPER = ("configuration_allie.py", "modeling_allie.py", "README.md")


def build(export, out):
    export, out = Path(export), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "model.safetensors"
    weights.unlink(missing_ok=True)
    try:
        os.link(export / "model.safetensors", weights)
    except OSError:
        shutil.copy(export / "model.safetensors", weights)
    config = json.loads((export / "config.json").read_text())
    config.pop("checkpoint", None)  # a path on the training cluster
    config |= dict(
        name="allie-2.0",
        model_type="allie",
        architectures=["AllieModel"],
        auto_map=dict(
            AutoConfig="configuration_allie.AllieConfig",
            AutoModel="modeling_allie.AllieModel",
        ),
    )
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    for name in CODE:
        shutil.copy(HERE / name, out / name)
    for name in WRAPPER:
        text = (HERE / "hf" / name).read_text().replace("GITHUB_URL", GITHUB)
        (out / name).write_text(text)
    return out


def upload(out, repo=REPO, private=True):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message="Allie 2.0")
    return f"https://huggingface.co/{repo}"


def main():
    p = argparse.ArgumentParser(prog="python -m allie.lichess.hub", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("export")
    b.add_argument("out")
    u = sub.add_parser("upload")
    u.add_argument("out")
    u.add_argument("--repo", default=REPO)
    u.add_argument("--public", action="store_true")
    a = p.parse_args()
    if a.command == "build":
        print(build(a.export, a.out))
    else:
        print(upload(a.out, a.repo, not a.public))


if __name__ == "__main__":
    main()
