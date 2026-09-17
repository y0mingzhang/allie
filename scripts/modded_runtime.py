"""Prepopulate Inductor's supported source-key cache for our pinned torch wheel.

The key was computed by torch_key() itself on the durable runtime. Checking the
wheel RECORD avoids re-reading thousands of torch files on every cold NFS node.
Other distributions use PyTorch's ordinary hashing path. This changes no kernels
or cache keys; it memoizes the immutable runtime's existing key.
"""
import hashlib
from pathlib import Path

import torch

PINNED_TORCH = '2.10.0+cu128'
RECORD_SHA256 = '29104f84f7727f6872373ba7e1dbf82ce194f95c8497d3583974ce0fac5c77af'
SOURCE_KEY = 'd4b79850ff54519c2421cfa2c38e39eb617533c190cf289c38e9df1a30c81751'


def prime_source_key():
    if torch.__version__ != PINNED_TORCH:
        return None
    from torch._inductor.codecache import torch_key
    record = Path(torch.__file__).parent.parent/f'torch-{PINNED_TORCH}.dist-info/RECORD'
    if not record.exists() or hashlib.sha256(record.read_bytes()).hexdigest() != RECORD_SHA256:
        return None
    try:
        torch_key.set(bytes.fromhex(SOURCE_KEY))
    except AssertionError:
        # A caller may already have computed or installed exactly this key.
        assert torch_key().hex() == SOURCE_KEY
    return SOURCE_KEY
