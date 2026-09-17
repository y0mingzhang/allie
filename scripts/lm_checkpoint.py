"""Atomic full-training checkpoints. Load only this project's trusted artifacts."""
import os
import random
import numpy as np
import torch

def rng_state():
    ns=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]))

def restore_rng(s):
    torch.set_rng_state(s['torch']);random.setstate(s['python'])
    ns=s['numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),*ns[2:]))
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def atomic_save(state,path):
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
