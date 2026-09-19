// Corrected soft backup with a configurable count normalization.
#include "diff_backup.cpp"
struct ScaledCount : Backup {
 ScaledCount(py::dict d,int budget,double scale):Backup(d,budget){
  if(!(scale>0)||!std::isfinite(scale))throw std::invalid_argument("positive scale required");
  std::vector<int> count(n,1);
  for(int i=n-1;i>=0;--i){for(int c=first[i];c>=0;c=next[c])count[i]+=count[c];lc[i]=std::log1p((count[i]-1.)/scale);}
 }
};
PYBIND11_MODULE(_allie_scaled_count,m){py::class_<ScaledCount>(m,"Backup").def(py::init<py::dict,int,double>()).def("reduce",&ScaledCount::reduce);}
