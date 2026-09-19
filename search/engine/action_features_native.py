"""Rule-derived action descriptors, no position evaluation or external engine."""
import hashlib,importlib,json,subprocess,sys,sysconfig
from pathlib import Path
import numpy as np
from .backup_native import ROOT

NAMES=['pawn','knight','bishop','rook','queen','king','takes_pawn','takes_knight','takes_bishop','takes_rook','takes_queen','takes_king',
       'check','castle','promotion','relative_rank','file_centrality','distance','log_reply_mobility','forced_reply']

def load():
    import pybind11
    source=Path(__file__).with_name('action_features.cpp')
    header=ROOT/'vendor/chess-library/include/chess.hpp'
    key=dict(source=hashlib.sha256(source.read_bytes()).hexdigest(),rules=hashlib.sha256(header.read_bytes()).hexdigest(),python=sys.version)
    folder=ROOT/'results/search-v1/runtime/action_features';folder.mkdir(exist_ok=True)
    target=folder/('_allie_action_features'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        tmp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-I'+str(header.parent),'-o',str(tmp)],check=True)
        tmp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_action_features')

def test(module):
    import chess
    from .native_board import MOVES
    from .sample_data import read
    d=read('aug-tune-expanded-v1')
    rows=d['rows'][::83][:180]
    got=module.encode([r['prefix'] for r in rows],[r['legal'] for r in rows],MOVES)
    expected=np.zeros_like(got)
    for i,r in enumerate(rows):
        b=chess.Board()
        for t in r['prefix'][11:]:b.push_uci(MOVES[t-378])
        assert set(r['legal'])=={378+MOVES.index(m.uci()) for m in b.legal_moves}
        for j,t in enumerate(r['legal']):
            m=chess.Move.from_uci(MOVES[t-378]);p=b.piece_type_at(m.from_square)
            x=expected[i,j];x[p-1]=1.
            if b.is_capture(m):victim=chess.PAWN if b.is_en_passant(m) else b.piece_type_at(m.to_square);x[6+victim-1]=1.
            x[12]=b.gives_check(m);x[13]=b.is_castling(m);x[14]=bool(m.promotion)
            ff,fr=chess.square_file(m.from_square),chess.square_rank(m.from_square)
            tf,tr=chess.square_file(m.to_square),chess.square_rank(m.to_square)
            x[15]=(tr if b.turn==chess.WHITE else 7-tr)/7
            x[16]=1-abs(tf-3.5)/3.5;x[17]=np.hypot(tf-ff,tr-fr)/np.sqrt(98)
            c=b.copy();c.push(m);nm=c.legal_moves.count();x[18]=np.log1p(nm);x[19]=nm==1
    np.testing.assert_allclose(got,expected,atol=1e-14)
    print('PASS rule features against independent python-chess:',len(rows),'positions',sum(len(r['legal']) for r in rows),'legal moves',flush=True)

if __name__=='__main__':test(load())
