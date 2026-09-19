// Continue a completed root-coverage forest without re-evaluating any node.
#include "threadforest.cpp"
struct GrowForest : ThreadForestTree {
    using ThreadForestTree::ThreadForestTree;
    void grow(const std::vector<int>& next_budgets) {
        if(!done()) throw std::invalid_argument("finish current phase before grow");
        if(next_budgets.size()!=roots.size()) throw std::invalid_argument("budget shape");
        for(int i=0;i<(int)roots.size();++i)
            if(next_budgets[i]<budgets[i]) throw std::invalid_argument("cannot shrink");
        int offset=0;
        for(int owner=0;owner<(int)roots.size();++owner) {
            int root=roots[owner],size=nodes[root].children.size();
            std::vector<double> weights(size);
            std::vector<int> counts(size,0);
            for(int j=0;j<size;++j) {
                double p=nodes[root].children[j].prior;
                weights[j]=std::sqrt(p*(1-p));
                if(branches[offset+j].next!=(int)branches[offset+j].pulls.size())
                    throw std::runtime_error("unfinished branch");
            }
            for(int pull=1;pull<=next_budgets[owner];++pull) {
                int best=0;double best_u=-std::numeric_limits<double>::infinity();
                for(int j=0;j<size;++j) {
                    double u=weights[j]/(1+counts[j]);
                    if(u>best_u){best_u=u;best=j;}
                }
                auto& branch=branches[offset+best];
                if(pull<=budgets[owner]) {
                    if(counts[best]>=(int)branch.pulls.size() || branch.pulls[counts[best]]!=pull)
                        throw std::runtime_error("quota prefix changed");
                } else {
                    branch.pulls.push_back(pull);remaining++;
                }
                counts[best]++;
            }
            offset+=size;
        }
        budgets=next_budgets;limit=*std::max_element(budgets.begin(),budgets.end());
    }
};
PYBIND11_MODULE(_allie_growforest,m) {
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    py::class_<GrowForest>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int>())
        .def("select",&GrowForest::next_handles).def("update",&GrowForest::update_fast)
        .def("grow",&GrowForest::grow).def("compact",&GrowForest::compact)
        .def("stats",&GrowForest::stats).def("snapshot",&GrowForest::snapshot)
        .def_property_readonly("evals",[](const GrowForest& x){return x.evals;})
        .def_property_readonly("done",&GrowForest::done);
}
