"""Causal board and imagined clock transitions; no labels or future sidecars."""
import numpy as np
from scipy.special import softmax
from .board import _load


def advance_boards(parent,tokens):
    _,table=_load();b=np.asarray(parent,np.uint8).copy();n=len(b);ix=np.arange(n)
    tokens=np.asarray(tokens);assert ((tokens>=378)&(tokens<2346)).all()
    fr,to,prom=table[tokens].T;white=b[:,64].astype(bool);piece=b[ix,fr].astype(int)
    assert (b[:,67]==1).all() and (piece>0).all() and ((piece<=6)==white).all()
    kind=(piece-1)%6+1;capture=b[ix,to].copy()
    ep=np.where(b[:,66]>0,b[:,66].astype(int)-1+np.where(white,40,16),-1)
    passant=(kind==1)&(to==ep)&(capture==0)&(fr%8!=to%8)
    b[ix[passant],to[passant]+np.where(white[passant],-8,8)]=0
    b[ix,fr]=0;b[ix,to]=np.where(prom>0,prom+np.where(white,0,6),piece)
    king=kind==6;b[king,65]&=np.where(white[king],12,3).astype(np.uint8)
    castle=king&(abs(to-fr)==2);right=to>fr
    rook_from=np.where(right,to+1,to-2);rook_to=(fr+to)//2
    b[ix[castle],rook_to[castle]]=b[ix[castle],rook_from[castle]];b[ix[castle],rook_from[castle]]=0
    for square,mask in [(0,253),(7,254),(56,247),(63,251)]:b[(fr==square)|(to==square),65]&=mask
    b[:,66]=np.where((kind==1)&(abs(to-fr)==16),to%8+1,0)
    b[:,64]=~white
    return b


def predicted_seconds(logits):
    centers=np.r_[np.arange(16),16*np.exp(np.arange(47)/7.06)]
    return softmax(np.asarray(logits,dtype=np.float64)[:,2350:2413],axis=1)@centers


def advance_clocks(parent,other_previous,lengths,increments,elapsed):
    """Third channel is this mover's PREVIOUS OWN think time, not last ply's time.

    Parent predicts move index length-11 (zero based). Each player's first move
    does not tick the clock, matching the frozen training sampler.
    """
    parent=np.asarray(parent);ply=np.asarray(lengths)-11;inc=np.asarray(increments)
    valid=(parent[:,0]>=0)&(parent[:,1]>=0)&(inc>=0)
    spent=np.minimum(np.maximum(np.rint(elapsed),0),np.maximum(parent[:,0],0))
    spent=np.where(ply<2,0,spent)
    after=np.maximum(0,parent[:,0]-spent)+np.where(ply<2,0,inc)
    child=np.stack((parent[:,1],after,np.where(ply+1>=4,other_previous,-1)),1)
    child[~valid]=-1
    next_previous=np.where(valid&(ply>=2),spent,-1)
    return child,next_previous


def root_other_previous(prefix,features,increment):
    m=len(prefix)-11
    if m<3 or increment<0:return -1.
    # Last opponent move's elapsed time is known from clocks in the prefix.
    before=features[-2,0];after=features[-1,1]
    return float(before-after+increment) if before>=0 and after>=0 and before-after+increment>=0 else -1.
