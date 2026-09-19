// Same audited NativeMCTS; only the query transport changes to owned handles.
#include "../engine/board.cpp"
struct HandleMCTS: NativeMCTS {
    using NativeMCTS::NativeMCTS;
    py::array_t<int> handles(){
        select();py::array_t<int> out({int(pending.size()),4});
        for(int i=0;i<int(pending.size());++i){int id=pending[i];int* p=out.mutable_data(i,0);
            p[0]=id;p[1]=nodes[id].parent;p[2]=nodes[id].move;p[3]=nodes[id].prefix.size();}
        return out;
    }
};
PYBIND11_MODULE(_ship_allie_handles,m){
    m.def("initialize",[](const std::vector<std::string>& v){moves=v;ids.clear();for(int i=0;i<int(v.size());++i)ids[v[i]]=378+i;});
    py::class_<NativeMCTS>(m,"Reference",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def_readwrite("first_prior",&NativeMCTS::first_prior).def_readwrite("preserve_depth",&NativeMCTS::preserve_depth)
        .def("select",&NativeMCTS::select).def("update",&NativeMCTS::update)
        .def("summaries",&NativeMCTS::summaries).def("stats",&NativeMCTS::stats);
    py::class_<HandleMCTS>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def_readwrite("first_prior",&HandleMCTS::first_prior).def_readwrite("preserve_depth",&HandleMCTS::preserve_depth)
        .def("select",&HandleMCTS::handles).def("update",&HandleMCTS::update)
        .def("summaries",&HandleMCTS::summaries).def("stats",&HandleMCTS::stats)
        .def_property_readonly("done",[](const HandleMCTS& x){return x.iteration>=x.limit && x.pending.empty();});
}
