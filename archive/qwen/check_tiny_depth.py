"""CPU depth generalization checks; 16-layer exact equivalence and causal gradients."""
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/recipe10x/tiny-scaling-v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--source');p.add_argument('--layers',type=int);p.add_argument('--width',type=int);p.add_argument('--out');a=p.parse_args()
    if a.worker:
        os.environ['ALLIE_PROJECT_ROOT']=str(ROOT);sys.path.insert(0,a.source)
        import torch,torch.distributed as dist
        from modded_medium import Config,create_model,make_context,core
        torch.set_num_threads(4);torch.manual_seed(42)
        dist.init_process_group('gloo',init_method='file://'+tempfile.mktemp(prefix='tiny-depth-gloo-'),rank=0,world_size=1)
        model=create_model(Config(width=a.width,layers=a.layers,head_dim=64,max_tokens=1024),device='cpu').float()
        def h(v):return hashlib.sha256(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        initial={n:h(v) for n,v in model.state_dict().items()}
        with torch.no_grad():
            for v in model.parameters():v.normal_(std=.03)
        model.split_embed=True
        core.args.block_size=1
        x=torch.randint(378,2346,(1,32));x[:,0]=2348;x[:,16]=2348;y=x.roll(-1,1)
        context=make_context(x,8,16,'dense');schedule=core.ForwardScheduleConfig(torch.ones(1),8,16)
        logits=model(x.flatten(),y.flatten(),context,schedule)
        logits.square().mean().backward()
        grads={n:h(v.grad) for n,v in model.named_parameters() if v.grad is not None}
        assert len(grads)==len(list(model.parameters()))
        assert all(torch.isfinite(v.grad).all() for v in model.parameters())
        with torch.no_grad():
            z=x.clone();z[:,1:16]=450
            changed=model(z.flatten(),y.flatten(),context,schedule)
            assert torch.equal(logits[:,16:],changed[:,16:]),'Cross-game leakage'
            z=x.clone();z[:,25:]=451
            changed=model(z.flatten(),y.flatten(),context,schedule)
            assert torch.equal(logits[:,:25],changed[:,:25]),'Future-token leakage'
        Path(a.out).write_text(json.dumps(dict(passed=True,parameters=sum(v.numel() for v in model.parameters()),initial=initial,
            output=h(logits),gradients=grads,causal_and_game_isolation=True,all_parameters_receive_finite_gradients=True),indent=2)+'\n')
        dist.destroy_process_group();return
    evidence=OUT/'cpu-check';evidence.mkdir(exist_ok=True)
    tasks=[('old16',ROOT/'results/recipe10x/schedule-quality-v1/source',16,128),
           ('new16',OUT/'source-ours',16,128),('new8',OUT/'source-ours',8,128),('new12',OUT/'source-ours',12,256)]
    for key,source,layers,width in tasks:
        with (evidence/(key+'.log')).open('w') as log:
            subprocess.run([sys.executable,__file__,'--worker','--source',str(source),'--layers',str(layers),'--width',str(width),'--out',str(evidence/(key+'.json'))],check=True,stdout=log,stderr=subprocess.STDOUT)
    old=json.loads((evidence/'old16.json').read_text());new=json.loads((evidence/'new16.json').read_text())
    assert old==new,'16-layer initialization/forward/gradients must remain exact'
    counts={'ours':{'128':json.loads((evidence/'new8.json').read_text())['parameters'],
                    '256':json.loads((evidence/'new12.json').read_text())['parameters'],'512':60296597},'qwen_recipe':{}}
    sys.path.insert(0,str(ROOT/'results/recipe10x/native-curvature-v1/native128-third'))
    from scaled_native_qwen import Shape
    for w,l in [(128,8),(256,12),(512,16)]:
        counts['qwen_recipe'][str(w)]=Shape(layers=l,width=w,ff={128:448,256:896,512:1728}[w],heads=w//128,kv_heads=max(1,w//256)).parameter_count()
    report=dict(passed=True,parameters=counts,old16_new16_initial_state_forward_all_gradients_bitwise_equal=True,
        depths8_12_causal_isolated_finite_all_parameter_gradients=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'cpu-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
