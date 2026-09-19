"""Expand the frozen pilot choice on ALL existing development fold-1 positions.
No parameter search. Sends batches to the already allocated persistent GPU oracle.
"""
import hashlib,json,sys,time,uuid,io,urllib.request
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
import chess
ROOT=Path(__file__).resolve().parents[1]/'results/search-v1'
OUT=ROOT/'confirmation';DATA=Path('/home/yimingz3/src/allie/data')
SOURCE=Path('/data/group_data/dei-group/yimingz3/allie/results/recipe10x/data-v1-round2/source-ours')
sys.path.insert(0,str(SOURCE))
from chess_vocab import MOVES,MOVE_ID

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def call(prefixes,name,columns=None):
    output=OUT/(name+'.logits.npz')
    if output.exists():
        with np.load(output) as z:
            a=z['logits']
            return a[:,columns] if columns is not None and a.shape[1]==2432 else a
    ready=json.loads((ROOT/'server-ready.json').read_text())
    token=(ROOT/'rpc-token').read_text().strip()
    request=dict(prefixes=prefixes)
    if columns is not None:request['columns']=columns
    req=urllib.request.Request(ready['url'],data=json.dumps(request).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=120) as response:a=np.load(io.BytesIO(response.read()),allow_pickle=False)
    tmp=output.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez(f,logits=a)
    tmp.replace(output)
    return a

def positions():
    seen=set()
    for split in ('dev','dev_expert'):
        games=[json.loads(s) for s in (DATA/f'{split}.jsonl').open()]
        for shard in range(4):
            p=DATA/f'prepared-{split}-4-{shard}'
            with np.load(p.with_suffix('.npz')) as z:meta={k:z[k] for k in ('game','ply','elos','raw_target','qids')}
            for tokens,start,end in json.loads(p.with_suffix('.json').read_text()):
                if start==end:continue
                game=games[int(meta['game'][start])]['game-id']
                if int(hashlib.sha256(game.encode()).hexdigest()[:8],16)%2!=1:continue
                board=chess.Board();current=0
                for j in range(start,end):
                    ply=int(meta['ply'][j]);elo=int(meta['elos'][j,0])
                    if (game,ply) in seen or (split=='dev_expert' and elo<2400):continue
                    seen.add((game,ply))
                    while current<ply:board.push(chess.Move.from_uci(MOVES[tokens[11+current]-378]));current+=1
                    legal=meta['qids'][j][meta['qids'][j]>=0].tolist()
                    assert {MOVE_ID[m.uci()] for m in board.legal_moves}==set(legal)
                    yield dict(game=game,ply=ply,elo=elo,split=split,prefix=tokens[:11+ply],
                        target=int(meta['raw_target'][j]),legal=legal,fen=board.fen())

def batch_score(rows,index,choice):
    dest=OUT/f'scores-{index:05d}.npz'
    if dest.exists():return
    r=call([p['prefix'] for p in rows],f'{index:05d}-root').astype(np.float64)
    cand=np.full((len(rows),8),-1,np.int16);child=np.full((len(rows),8),np.nan);where=[];prefix=[]
    for i,p in enumerate(rows):
        legal=np.array(p['legal']);moves=legal[np.argsort(r[i,legal])[::-1][:8]];cand[i,:len(moves)]=moves
        for j,t in enumerate(moves):
            board=chess.Board(p['fen']);board.push(chess.Move.from_uci(MOVES[int(t)-378]))
            if board.is_checkmate():child[i,j]=1
            elif board.is_stalemate() or board.is_insufficient_material():child[i,j]=.5
            else:prefix.append(p['prefix']+[int(t)]);where.append((i,j))
    if prefix:
        logits=call(prefix,f'{index:05d}-child',columns=[2413,2414,2415]).astype(np.float64)
        value=1-softmax(logits,axis=1)@np.array([1,.5,0])
        for (i,j),v in zip(where,value):child[i,j]=v
    rootvalue=softmax(r[:,2413:2416],axis=1)@np.array([1,.5,0])
    move=r[:,378:2346];legal=np.zeros(move.shape,bool);adv=np.zeros(move.shape)
    for i,p in enumerate(rows):
        legal[i,np.array(p['legal'])-378]=True;valid=cand[i]>=378
        adv[i,cand[i,valid]-378]=child[i,valid]-rootvalue[i]
    target=np.array([p['target']-378 for p in rows]);expert=np.array([p['elo']>=2400 for p in rows]); ar=np.arange(len(rows))
    specs=[dict(temperature=1),dict(temperature=1,legal=True),choice['calibration'],choice['selected']]
    losses=[];correct=[]
    for par in specs:
        enabled=expert if par.get('expert_only') else np.ones(len(rows),bool)
        x=move/np.where(enabled,par['temperature'],1.)[:,None]
        gate=softmax(r[:,2350:2413],axis=1)[:,6:].sum(1) if par.get('gate')=='time' else np.ones(len(rows))
        x+=par.get('beta',0)*(enabled*gate)[:,None]*adv
        if par.get('legal'):x=np.where(legal | ~enabled[:,None],x,-np.inf)
        losses.append(logsumexp(x,axis=1)-x[ar,target]);correct.append(x.argmax(1)==target)
    tmp=dest.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez(f,nll=np.stack(losses,1),correct=np.stack(correct,1),expert=expert,
        game=np.array([p['game'] for p in rows]),ply=np.array([p['ply'] for p in rows]))
    tmp.replace(dest)

def main():
    OUT.mkdir(exist_ok=True)
    choice=json.loads((ROOT/'selected.json').read_text());plan=OUT/'plan.json'
    manifest=dict(choice=choice,scope='All existing prepared dev/dev_expert fold1; fixed settings selected on pilot fold0; excludes test/golden',
        checkpoint=json.loads((ROOT/'server-ready.json').read_text())['checkpoint_sha256'])
    if plan.exists():assert json.loads(plan.read_text())==manifest
    else:plan.write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];index=0;count=0;start=time.monotonic()
    for p in positions():
        rows.append(p)
        if len(rows)==128:
            batch_score(rows,index,choice);count+=len(rows);rows=[];index+=1
            if index%20==0:print('confirmed positions',count,'seconds',time.monotonic()-start,flush=True)
    if rows:batch_score(rows,index,choice);count+=len(rows)
    arrays=[]
    for p in sorted(OUT.glob('scores-*.npz')):
        with np.load(p) as z:arrays.append({k:z[k] for k in z.files})
    data={k:np.concatenate([z[k] for z in arrays]) for k in arrays[0]};summary={}
    names=['canonical_raw','legal','calibrated','shallow']
    for j,name in enumerate(names):
        summary[name]=dict(dev_ce=float(data['nll'][:,j].mean()),expert_dev_ce=float(data['nll'][data['expert'],j].mean()),
            accuracy=float(data['correct'][:,j].mean()),expert_accuracy=float(data['correct'][data['expert'],j].mean()))
    _,inverse=np.unique(data['game'],return_inverse=True);ng=int(inverse.max())+1;rng=np.random.default_rng(823)
    ci={}
    for baseline in (0,2):
        delta=data['nll'][:,3]-data['nll'][:,baseline];samples=[]
        for _ in range(1000):
            w=rng.poisson(1,ng)[inverse];e=data['expert']
            samples.append([np.sum(w*delta)/max(1,w.sum()),np.sum(w[e]*delta[e])/max(1,w[e].sum())])
        ci[names[baseline]]=np.quantile(samples,[.025,.975],axis=0).tolist()
    report=dict(stage='Expanded unchanged development confirmation, NOT golden evaluation',positions=count,
        expert_positions=int(data['expert'].sum()),games=ng,methods=summary,paired_delta_ce_95pct=ci,
        selected=choice['selected'],elapsed_seconds=time.monotonic()-start)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
