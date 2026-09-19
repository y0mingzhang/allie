"""Build the small native rules adapter in our private runtime directory."""
import subprocess
import sysconfig
from pathlib import Path
import pybind11

root=Path(__file__).resolve().parents[2]
output=root/'results/search-v1/runtime/native'
output.mkdir(exist_ok=True)
target=output/('_allie_board'+sysconfig.get_config_var('EXT_SUFFIX'))
temporary=target.with_suffix('.new')
cmd=['c++','-O3','-std=c++17','-shared','-fPIC',str(Path(__file__).with_name('board.cpp')),
     '-I'+pybind11.get_include(),'-I'+sysconfig.get_path('include'),
     '-I'+str(root/'vendor/chess-library/include'),
     '-o',str(temporary)]
subprocess.run(cmd,check=True)
temporary.replace(target)  # Never truncate a library mapped by a resident runner.
print(output)
