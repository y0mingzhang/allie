"""Export a distributed native checkpoint, preserving every tensor payload."""
import argparse, ast, hashlib, json, os, re
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from transformers import Qwen3ForCausalLM
from scaled_native_qwen import Shape
ROOT=Path('/home/yimingz3/src/allie')
BASE=ROOT/'results/recipe10x/qwen-reproduction'
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def mapping(config, expected_parameters):
    # Read the exact pure-name conversion method from the recorded source;
    # avoid importing CUDA-only model kernels into this CPU export process.
    source = BASE/'source-v1/picotron/picotron/checkpoint.py'
    manifest = json.loads((BASE/'source-v1/manifest.json').read_text())
    assert digest(source)==manifest['files']['picotron/picotron/checkpoint.py']
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InitializationManager')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='convert_safetensors_to_hf_name')
    namespace = {'re':re}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    convert = namespace['convert_safetensors_to_hf_name']
    with torch.device('meta'):
        model = Qwen3ForCausalLM(config)
    assert sum(p.numel() for p in model.parameters())==expected_parameters
    shapes = {k:tuple(v.shape) for k,v in model.state_dict().items()}
    names = {name:convert(None,name) for name in shapes}
    assert len(names)==len(set(names.values()))==len(shapes)
    return names,shapes


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    checkpoint=Path(a.checkpoint)
    pointer=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert pointer['format']=='native-distributed-pointer-v1'
    directory=(checkpoint.parent/pointer['directory']).resolve()
    assert directory.is_relative_to(checkpoint.parent.resolve()/'checkpoints')
    model_file=directory/'model.pt'
    state=torch.load(model_file,map_location='cpu',weights_only=False,mmap=True)
    assert state['format']=='native-distributed-model-v1' and state['step']==pointer['step']
    world=state['metadata']['world_size'];rank_paths=[]
    for rank in range(world):
        path=directory/f'rank{rank}.pt'
        local=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert local['rank']==rank and local['world_size']==world and local['step']==state['step']
        if rank==0:state['data']=local['data']
        else:assert local['data']==state['data']
        rank_paths.append(path)
        del local

    shape=Shape(**state['metadata']['shape']);cfg=shape.hf_config();n=shape.parameter_count()
    assert state['metadata']['parameters']==n
    assert state['tokens']==state['step']*state['metadata']['recipe']['global_rows']*1024
    assert state['data']['seen']*1024==state['tokens']
    names,shapes=mapping(cfg,n)
    assert set(names.values())==set(state['model'])
    weights={hf:state['model'][native].contiguous() for hf,native in names.items()}
    for name,value in weights.items():assert tuple(value.shape)==shapes[name] and value.dtype==torch.bfloat16
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cfg.save_pretrained(out)
    save_file(weights,str(out/'model.safetensors'),metadata={'format':'pt'})
    restored=load_file(str(out/'model.safetensors'))
    assert restored.keys()==weights.keys()
    for name in weights:assert torch.equal(restored[name],weights[name]),name
    identity=dict(format='scaled-native-export-v1',path=str(out.resolve()),files_sha256={name:digest(out/name) for name in ('config.json','model.safetensors')},
        source_checkpoint=str(checkpoint.resolve()),source_checkpoint_sha256=digest(checkpoint),source_metadata=state['metadata'],step=state['step'],tokens=state['tokens'],
        parameters=n,tensors=len(names),exporter_sha256=digest(__file__),all_tensor_payloads_exact=True,final_test_accessed=False,
        original_dataset_revision='20a899ddf344ccaea74e273509a60e5a511125f8',scope='Bijective native-to-HF names only; exact roundtrip tensor equality')
    identity.update(distributed_checkpoint=True,source_model_file=str(model_file),source_model_sha256=digest(model_file),
        source_rank_state_sha256={p.name:digest(p) for p in rank_paths},world_size=world)
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({k:v for k,v in identity.items() if k!='source_metadata'}))
if __name__=='__main__':main()
