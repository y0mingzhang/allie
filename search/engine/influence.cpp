// Root coverage plus value-influence allocation below root.
// mix<0 is the exact coverage/PUCT control; mix1 uses only policy quotas.
#include "compact.cpp"

struct InfluenceTree : CompactTree {
    int fpu_mode;double reduction; bool use_soft; double tau,mix; std::vector<double> values;
    InfluenceTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                  std::vector<int> sims,std::vector<double> cpuct,int fpu,double penalty,double mixture,double temperature)
        :CompactTree(prefixes,scores,std::move(sims),std::move(cpuct)),fpu_mode(fpu),reduction(penalty),use_soft(mixture>=0 && mixture<1),tau(temperature),mix(mixture){
        if(fpu<0 || fpu>2 || penalty<0 || mix < -1 || mix > 1 || !std::isfinite(mix))throw std::invalid_argument("FPU config");
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
                double hi=-std::numeric_limits<double>::infinity(),softmass=0.,psum=0.;
                if(use_soft && nodes[id].parent>=0){
                    for(const auto& e:nodes[id].children){double q=e.child<0?-nodes[id].bootstrap:-values[e.child];hi=std::max(hi,q);}
                    for(const auto& e:nodes[id].children){double q=e.child<0?-nodes[id].bootstrap:-values[e.child];softmass+=e.prior*std::exp((q-hi)/tau);psum+=e.prior;}
                }
                int best=-1;double best_u=-std::numeric_limits<double>::infinity();
                for(size_t j=0;j<nodes[id].children.size();++j){
                    auto& edge=nodes[id].children[j];
                    int n=edge.child<0?0:nodes[edge.child].n;
                    double q=n?(use_soft?-values[edge.child]:nodes[edge.child].w/n):unseen;
                    double u=q+factor*edge.prior/(1+n);
                    if(nodes[id].parent<0)u=std::sqrt(edge.prior*(1-edge.prior))/(1+n);
                    else if(mix==1)u=edge.prior/(1+n);
                    else if(use_soft){
                        double value=edge.child<0?-nodes[id].bootstrap:-values[edge.child];
                        double softp=edge.prior*std::exp((value-hi)/tau)/softmass;
                        u=((1-mix)*softp+mix*edge.prior/psum)/(1+n);
                    }
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

PYBIND11_MODULE(_allie_influence,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<InfluenceTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int,double,double,double>())
        .def("select",&InfluenceTree::next_variant).def("update",&InfluenceTree::update_soft)
        .def("exported",&InfluenceTree::exported)
        .def_property_readonly("visits",[](const InfluenceTree& x){std::vector<int> n;for(const auto& a:x.nodes)n.push_back(a.n);return n;})
        .def("snapshot",&InfluenceTree::snapshot).def("backups",&InfluenceTree::backups)
        .def("compact",&InfluenceTree::compact).def("stats",&InfluenceTree::stats)
        .def_property_readonly("values",[](const InfluenceTree& x){return x.values;})
        .def_property_readonly("evals",[](const InfluenceTree& x){return x.evals;});
}
