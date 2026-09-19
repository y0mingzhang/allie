"""Create a private serving export; the training checkpoint stays read-only."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file
from .model import load_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    model,source=load_checkpoint(a.checkpoint)
    assert not model.config.clock and not model.config.elo, 'Side-channel request API is not implemented yet'
    config=dict(model_type='allie_chess',allie=vars(model.config),
        auto_map={'AutoConfig':'configuration_allie.AllieConfig'},architectures=['AllieForCausalLM'],
        torch_dtype='bfloat16')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    shutil.copy(Path(__file__).with_name('configuration_allie.py'),out/'configuration_allie.py')
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
    with source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(checkpoint=str(source),sha256=digest),indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
