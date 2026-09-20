// Audited repaired Allie, extended to reuse an exact common simulation prefix.
#include "../engine/board.cpp"
struct RoutedMCTS: NativeMCTS {
 using NativeMCTS::NativeMCTS;
 void grow(const std::vector<int>& next){
  if(!pending.empty() || iteration!=limit || next.size()!=budgets.size())throw std::invalid_argument("unfinished or wrong shape");
  for(size_t i=0;i<next.size();++i){
   if(next[i]<budgets[i])throw std::invalid_argument("cannot shrink");
   // Native iteration is shared: every growing root must have reached it.
   if(next[i]>budgets[i] && budgets[i]!=iteration)throw std::invalid_argument("unequal growing prefix");
  }
  budgets=next;limit=*std::max_element(budgets.begin(),budgets.end());
 }
 py::array_t<int> handles(){
  select();py::array_t<int> out({int(pending.size()),4});
  for(int i=0;i<int(pending.size());++i){int id=pending[i];int* p=out.mutable_data(i,0);p[0]=id;p[1]=nodes[id].parent;p[2]=nodes[id].move;p[3]=nodes[id].prefix.size();}
  return out;
 }
};
PYBIND11_MODULE(_allie_time_router,m){
 m.def("initialize",[](const std::vector<std::string>& v){moves=v;ids.clear();for(int i=0;i<int(v.size());++i)ids[moves[i]]=378+i;});
 py::class_<RoutedMCTS>(m,"Tree",py::module_local())
  .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
  .def_readwrite("first_prior",&RoutedMCTS::first_prior).def_readwrite("preserve_depth",&RoutedMCTS::preserve_depth)
  .def("grow",&RoutedMCTS::grow).def("select",&RoutedMCTS::handles).def("update",&RoutedMCTS::update)
  .def("summaries",&RoutedMCTS::summaries).def("stats",&RoutedMCTS::stats)
  .def_property_readonly("done",[](const RoutedMCTS& x){return x.iteration==x.limit&&x.pending.empty();});
}
