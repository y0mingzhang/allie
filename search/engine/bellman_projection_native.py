"""Interpreter-specific native output processor; isolated from CPU audit builds."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import sysconfig


def load():
    import pybind11
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__).with_name(s) for s in ('bellman_projection.cpp','diff_backup.cpp')]
    folder=root/'results/search-v1/runtime/bellman-serving';folder.mkdir(exist_ok=True)
    key={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}|{'python':sys.version}
    target=folder/('_allie_bellman_projection'+sysconfig.get_config_var('EXT_SUFFIX'));stamp=folder/'build.json'
    if not target.exists() or not stamp.exists() or json.loads(stamp.read_text())!=key:
        temp=target.with_suffix('.new')
        subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(sources[0]),
            '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),'-o',str(temp)],check=True)
        temp.replace(target);stamp.write_text(json.dumps(key))
    sys.path.insert(0,str(folder));return importlib.import_module('_allie_bellman_projection')
