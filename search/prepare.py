"""Index existing prepared dev splits; no new evaluation games or filtering."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import chess
BASE=Path('/data/group_data/dei-group/yimingz3/allie')
SOURCE=BASE/'results/recipe10x/data-v1-round2/source-ours'
sys.path.insert(0,str(SOURCE))
from chess_vocab import MOVES,MOVE_ID
OUT=Path(__file__).resolve().parents[1]/'results/search-v1'
DATA=Path('/home/yimingz3/src/allie/data')

def main():
    dest=OUT/'dev.json'
    assert not dest.exists(),'Prepared inputs are immutable'
    rng=np.random.default_rng(823177); positions=[]; hashes={}; seen=set()
    for split in ('dev','dev_expert'):
        games=[json.loads(s) for s in (DATA/f'{split}.jsonl').open()]
        pool=[]
        for shard in range(4):
            p=DATA/f'prepared-{split}-4-{shard}'
            for suffix in ('.json','.npz'):
                with p.with_suffix(suffix).open('rb') as f:hashes[p.name+suffix]=hashlib.file_digest(f,'sha256').hexdigest()
            with np.load(p.with_suffix('.npz')) as z:
                # Do not load unrelated old engine predictions/history features.
                meta={k:z[k] for k in ('game','ply','elos','raw_target','qids')}
            prompts=json.loads(p.with_suffix('.json').read_text())
            for tokens,start,end in prompts:
                for j in range(start,end):
                    gid=int(meta['game'][j]);ply=int(meta['ply'][j]); g=games[gid]
                    key=(g['game-id'],ply)
                    if key in seen: continue
                    elo=int(meta['elos'][j,0])
                    if split=='dev_expert' and elo<2400: continue
                    seen.add(key)
                    # Existing prepared prompts define exact allowed positions.
                    base,inc=map(int,g['time-control'].split('+'))
                    duration=base+40*inc
                    fmt=0 if duration<180 else 1 if duration<480 else 2 if duration<1500 else 3
                    cell=fmt*4+int(np.searchsorted([1400,2000,2400],elo,side='right'))
                    pool.append(dict(game=g['game-id'],cell=cell,ply=ply,
                        prefix=tokens[:11+ply],target=int(meta['raw_target'][j]),
                        legal=meta['qids'][j][meta['qids'][j]>=0].tolist(),
                        fold=int(hashlib.sha256(g['game-id'].encode()).hexdigest()[:8],16)%2,
                        split=split,game_index=gid))
        # First screen is a deterministic subsample of existing eval positions.
        take=rng.choice(len(pool),min(1024,len(pool)),replace=False)
        for j in take:
            p=pool[int(j)];board=chess.Board()
            for t in p['prefix'][11:]: board.push(chess.Move.from_uci(MOVES[t-378]))
            actual={MOVE_ID[m.uci()] for m in board.legal_moves}
            assert actual==set(p['legal']),(p['game'],p['ply'],'legal alignment')
            assert p['target'] in actual
            p['fen']=board.fen();positions.append(p)
    dest.write_text(json.dumps(dict(positions=positions,input_sha256=hashes,
        protocol='Deterministic 1024-position subsample each of existing dev and expert-dev; unchanged prepared target/legality filters. Game-hash folds. Development only, not a new golden eval.'))+'\n')
    print('Wrote',len(positions),'positions; cells',np.bincount([p['cell'] for p in positions],minlength=16).tolist())
if __name__=='__main__':main()
