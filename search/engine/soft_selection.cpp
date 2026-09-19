// Separate experiment: select using current soft Bellman values rather than rollout means.
// Disabled mode is exactly the FPU baseline. Outputs still use the audited backups().
#include "compact.cpp"

struct SoftSelectionTree : CompactTree {
    int fpu_mode;double reduction; bool use_soft; double tau; std::vector<double> values;
    SoftSelectionTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                  std::vector<int> sims,std::vector<double> cpuct,int fpu,double penalty,bool soft,double temperature)
        :CompactTree(prefixes,scores,std::move(sims),std::move(cpuct)),fpu_mode(fpu),reduction(penalty),use_soft(soft),tau(temperature){
        if(fpu<0 || fpu>2 || penalty<0)throw std::invalid_argument("FPU config");
        if(!(tau>0) || !std::isfinite(tau))throw std::invalid_argument("soft temperature");
        values.resize(nodes.size());for(int id:roots)refresh(id);
    }
    void refresh(int id){
        values.resize(nodes.size());
        while(id>=0){
            const auto& node=nodes[id];double terminal=node.position->outcome();
            if(terminal>=0)values[id]=terminal==.5?0.:-1.;
            else if(node.children.empty())values[id]=-node.bootstrap;
            else{
                double hi=-std::numeric_limits<double>::infinity(),mass=0.,psum=0.;
                for(const auto& e:node.children){double q=e.child<0?-node.bootstrap:-values[e.child];hi=std::max(hi,q);}
                for(const auto& e:node.children){double q=e.child<0?-node.bootstrap:-values[e.child];mass+=e.prior*std::exp((q-hi)/tau);psum+=e.prior;}
                values[id]=hi+tau*std::log(mass/psum);
            }
            id=node.parent;
        }
    }
    void update_soft(Scores scores){
        auto changed=pending;NativeMCTS::update(scores);
        if(use_soft)for(int id:changed)refresh(id);
    }
    std::vector<std::vector<int>> next_variant(){
        if(!pending.empty())throw std::runtime_error("update pending predictions first");
        if(iteration>=limit)throw std::runtime_error("already finished");
        std::vector<std::vector<int>> prefixes;
        for(size_t i=0;i<roots.size();++i){
            if(budgets[i]<=iteration)continue;
            int id=roots[i],depth_limit=std::min(max_search_depth,1025-int(nodes[id].prefix.size()));
            while(!nodes[id].children.empty() && nodes[id].depth<depth_limit){
                double factor=(std::log((nodes[id].n+19652.+1)/19652.)+cp[i])*std::sqrt(double(first_prior?std::max(nodes[id].n,1):nodes[id].n));
                double unseen=0.;
                if(fpu_mode){
                    unseen=fpu_mode==2 && nodes[id].n ? -nodes[id].w/nodes[id].n : -nodes[id].bootstrap;
                    if(reduction){
                        double visited_mass=0.;
                        for(const auto& e:nodes[id].children)if(e.child>=0 && nodes[e.child].n)visited_mass+=e.prior;
                        unseen=std::max(-1.,unseen-reduction*std::sqrt(visited_mass));
                    }
                }
                int best=-1;double best_u=-std::numeric_limits<double>::infinity();
                for(size_t j=0;j<nodes[id].children.size();++j){
                    auto& edge=nodes[id].children[j];
                    int n=edge.child<0?0:nodes[edge.child].n;
                    double q=n?(use_soft?-values[edge.child]:nodes[edge.child].w/n):unseen;
                    double u=q+factor*edge.prior/(1+n);
                    if(u>best_u){best_u=u;best=int(j);}
                }
                if(nodes[id].children[best].child<0){
                    int move=nodes[id].children[best].move;double prior=nodes[id].children[best].prior;
                    int child=add(id,move,prior);nodes[id].children[best].child=child;
                }
                id=nodes[id].children[best].child;materialize(id);
            }
            max_depth=std::max(max_depth,nodes[id].depth);
            double outcome=nodes[id].position->outcome();
            if(outcome>=0){backup(id,outcome==.5?0.:1.);terminal_visits++;if(use_soft)refresh(id);}
            else if(preserve_depth && !nodes[id].children.empty() && nodes[id].depth>=depth_limit){backup(id,nodes[id].bootstrap);depth_visits++;}
            else{pending.push_back(id);prefixes.push_back(nodes[id].prefix);prefix_tokens+=nodes[id].prefix.size();}
        }
        iteration++;if(!pending.empty())requests++;
        evaluated+=pending.size();born.resize(nodes.size(),iteration);
        for(int id:pending){while(nodes[id].parent>=0)id=nodes[id].parent;evals[root_index.at(id)]++;}
        return prefixes;
    }
};

PYBIND11_MODULE(_allie_softselection,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<SoftSelectionTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int,double,bool,double>())
        .def("select",&SoftSelectionTree::next_variant).def("update",&SoftSelectionTree::update_soft)
        .def("snapshot",&SoftSelectionTree::snapshot).def("backups",&SoftSelectionTree::backups)
        .def("compact",&SoftSelectionTree::compact).def("stats",&SoftSelectionTree::stats)
        .def_property_readonly("values",[](const SoftSelectionTree& x){return x.values;})
        .def_property_readonly("evals",[](const SoftSelectionTree& x){return x.evals;});
}
