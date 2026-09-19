// Exact parallel scheduling of independent root-action subtrees.
// Quotas depend on root priors only; preserve each branch's logical simulation
// order and node birth time so prefix-budget reductions remain well defined.
#include "coverage.cpp"

struct ForestBranch {
    int owner,edge,next=0;
    std::vector<int> pulls;
};
struct ForestTree : CompactTree {
    std::vector<ForestBranch> branches;
    int remaining=0,rounds=0;
    ForestTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
               std::vector<int> sims,std::vector<double> cpuct)
        :CompactTree(prefixes,scores,std::move(sims),std::move(cpuct)){
        for(int owner=0;owner<(int)roots.size();++owner){
            int root=roots[owner],offset=branches.size();
            std::vector<double> weights;
            for(int j=0;j<(int)nodes[root].children.size();++j){
                double p=nodes[root].children[j].prior;
                weights.push_back(std::sqrt(p*(1-p)));
                branches.push_back(ForestBranch{owner,j,0,{}});
            }
            std::vector<int> counts(weights.size(),0);
            for(int pull=1;pull<=budgets[owner];++pull){
                int best=0;double best_u=-std::numeric_limits<double>::infinity();
                for(int j=0;j<(int)weights.size();++j){
                    double u=weights[j]/(1+counts[j]);
                    if(u>best_u){best_u=u;best=j;}
                }
                branches[offset+best].pulls.push_back(pull);counts[best]++;
                remaining++;
            }
        }
    }
    int child(int parent,int edge_index,int birth){
        if(nodes[parent].children[edge_index].child<0){
            int move=nodes[parent].children[edge_index].move;
            double prior=nodes[parent].children[edge_index].prior;
            int id=add(parent,move,prior);
            nodes[parent].children[edge_index].child=id;
            born.push_back(birth);
            if(born.size()!=nodes.size())throw std::runtime_error("birth accounting");
        }
        int id=nodes[parent].children[edge_index].child;materialize(id);return id;
    }
    bool done()const{return remaining==0 && pending.empty();}
    std::vector<std::vector<int>> next_forest(){
        if(!pending.empty())throw std::runtime_error("update pending predictions first");
        if(done())throw std::runtime_error("already finished");
        std::vector<std::vector<int>> prefixes;
        for(auto& branch:branches){
            if(branch.next==(int)branch.pulls.size())continue;
            int owner=branch.owner,birth=branch.pulls[branch.next++];
            int root=roots[owner],depth_limit=std::min(max_search_depth,1025-int(nodes[root].prefix.size()));
            int id=child(root,branch.edge,birth);
            while(!nodes[id].children.empty() && nodes[id].depth<depth_limit){
                double factor=(std::log((nodes[id].n+19652.+1)/19652.)+cp[owner])*std::sqrt(double(std::max(nodes[id].n,1)));
                int best=-1;double best_u=-std::numeric_limits<double>::infinity();
                for(int j=0;j<(int)nodes[id].children.size();++j){
                    auto& e=nodes[id].children[j];int n=e.child<0?0:nodes[e.child].n;
                    double q=n?nodes[e.child].w/n:0.;
                    double u=q+factor*e.prior/(1+n);
                    if(u>best_u){best_u=u;best=j;}
                }
                id=child(id,best,birth);
            }
            max_depth=std::max(max_depth,nodes[id].depth);
            double outcome=nodes[id].position->outcome();
            if(outcome>=0){backup(id,outcome==.5?0.:1.);terminal_visits++;}
            else if(!nodes[id].children.empty() && nodes[id].depth>=depth_limit){backup(id,nodes[id].bootstrap);depth_visits++;}
            else{pending.push_back(id);prefixes.push_back(nodes[id].prefix);prefix_tokens+=nodes[id].prefix.size();evals[owner]++;}
            remaining--;
        }
        rounds++;iteration=done()?limit:rounds;
        if(!pending.empty())requests++;
        evaluated+=pending.size();return prefixes;
    }
    std::vector<int64_t> prefix_evals(int budget)const{
        if(!done())throw std::runtime_error("finish before querying prefix counts");
        std::vector<int64_t> out(roots.size(),0);std::vector<int> owners(nodes.size(),-1);
        for(int i=0;i<(int)roots.size();++i)owners[roots[i]]=i;
        for(int id=0;id<(int)nodes.size();++id){
            if(nodes[id].parent<0)continue;
            owners[id]=owners[nodes[id].parent];
            if(born[id]<=budget && nodes[id].position->outcome()<0)out[owners[id]]++;
        }
        return out;
    }
};

PYBIND11_MODULE(_allie_forest,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<ForestTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def("select",&ForestTree::next_forest).def("update",&ForestTree::update)
        .def("snapshot",&ForestTree::snapshot).def("backups",&ForestTree::backups)
        .def("compact",&ForestTree::compact).def("stats",&ForestTree::stats)
        .def("prefix_evals",&ForestTree::prefix_evals)
        .def_readwrite("max_search_depth",&ForestTree::max_search_depth)
        .def_property_readonly("evals",[](const ForestTree& x){return x.evals;})
        .def_property_readonly("done",&ForestTree::done);
    py::class_<CoverageTree>(m,"Reference",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int,double,double>())
        .def("select",&CoverageTree::next_variant).def("update",&CoverageTree::update)
        .def("snapshot",&CoverageTree::snapshot).def("backups",&CoverageTree::backups)
        .def("compact",&CoverageTree::compact).def("stats",&CoverageTree::stats)
        .def_readwrite("max_search_depth",&CoverageTree::max_search_depth)
        .def_property_readonly("evals",[](const CoverageTree& x){return x.evals;});
}
