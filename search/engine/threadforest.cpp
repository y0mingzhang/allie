// Expand independent leaves concurrently; preserve exact serial backup order.
#include <exception>
#include "handleforest.cpp"
struct ThreadForestTree : HandleForestTree {
    int threads;
    ThreadForestTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                     std::vector<int> sims,std::vector<double> cpuct,int nthreads)
        :HandleForestTree(prefixes,scores,std::move(sims),std::move(cpuct)),threads(nthreads){
        if(threads<1 || threads>8)throw std::invalid_argument("threads must be 1..8");
    }
    void update_fast(Scores scores){
        if(scores.ndim()!=2 || scores.shape(0)!=(int)pending.size() || scores.shape(1)!=2432)
            throw std::invalid_argument("leaf dimensions");
        int count=pending.size();const float* z=scores.data();
        std::vector<double> values(count);std::exception_ptr error;
        #pragma omp parallel for num_threads(threads) if(count>=128 && threads>1) schedule(static)
        for(int i=0;i<count;++i){
            try {values[i]=expand(pending[i],z+2432*i);}
            catch(...){
                #pragma omp critical
                {if(!error)error=std::current_exception();}
            }
        }
        if(error)std::rethrow_exception(error);
        for(int i=0;i<count;++i)backup(pending[i],values[i]);
        pending.clear();
    }
};
PYBIND11_MODULE(_allie_threadforest,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<ThreadForestTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int>())
        .def("select",&ThreadForestTree::next_handles).def("update",&ThreadForestTree::update_fast)
        .def("snapshot",&ThreadForestTree::snapshot).def("backups",&ThreadForestTree::backups)
        .def("compact",&ThreadForestTree::compact).def("stats",&ThreadForestTree::stats)
        .def("prefix_evals",&ThreadForestTree::prefix_evals)
        .def_property_readonly("evals",[](const ThreadForestTree& x){return x.evals;})
        .def_property_readonly("done",&ThreadForestTree::done);
}
