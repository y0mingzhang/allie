"""Run the pinned SGLang setup function without importing its serving stack.

We execute the exact, hash-checked function AST from the installed Apache-2.0
SGLang0.5.9 engine.py. Its globals are explicit; tokenizer/scheduler/image imports
are unnecessary for the standalone ModelRunner. A dependency update fails closed.
"""
import ast
import hashlib
import importlib.util
import logging
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import time

FUNCTION_SHA256='d9c432e8c99c1df678f40166dcb703d062e35f657b686f88876fada65543c202'


def setup(server_args):
    from sglang.srt.utils import (assert_pkg_version,get_bool_env_var,is_cuda,
                                  kill_process_tree,set_prometheus_multiproc_dir,set_ulimit)
    package=Path(importlib.util.find_spec('sglang').origin).parent
    source=package/'srt/entrypoints/engine.py';text=source.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_set_envs_and_config')
    assert hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()==FUNCTION_SHA256,'Review changed SGLang setup before running'
    scope=dict(os=os,time=time,random=random,signal=signal,mp=mp,
               ServerArgs=type(server_args),logger=logging.getLogger(__name__),_is_cuda=is_cuda(),
               assert_pkg_version=assert_pkg_version,get_bool_env_var=get_bool_env_var,
               kill_process_tree=kill_process_tree,set_prometheus_multiproc_dir=set_prometheus_multiproc_dir,
               set_ulimit=set_ulimit)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    scope['_set_envs_and_config'](server_args)
