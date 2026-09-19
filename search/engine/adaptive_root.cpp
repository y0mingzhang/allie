// Root allocation follows the diagonal CE curvature of the updated policy.
// Each block freezes its quota, retaining independent subtrees within the block.
#include "threadforest.cpp"
struct AdaptiveRootTree : ThreadForestTree {
    double mixture;int block,stages=0;
    std::vector<double> beta,values;
    std::vector<unsigned char> dirty;
    std::vector<int> dirty_ids;
    AdaptiveRootTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                     std::vector<int> sims,std::vector<double> cpuct,int threads,
                     double mix,std::vector<double> inverse_temp,int block_size)
        :ThreadForestTree(prefixes,scores,std::move(sims),std::move(cpuct),threads),
         mixture(mix),block(block_size),beta(std::move(inverse_temp)){
        if(!(mixture>=0 && mixture<=1) || block<=0 || beta.size()!=roots.size())
            throw std::invalid_argument("adaptive root settings");
        for(double x:beta)if(!(x>=0 && x<=40))throw std::invalid_argument("inverse temperature");
        values.resize(nodes.size());dirty.resize(nodes.size(),0);
        for(int id:roots)values[id]=-nodes[id].bootstrap;
        plan_stage();
    }
    void mark(int id){
        values.resize(nodes.size());dirty.resize(nodes.size(),0);
        while(id>=0 && !dirty[id]){dirty[id]=1;dirty_ids.push_back(id);id=nodes[id].parent;}
    }
    void refresh(){
        std::sort(dirty_ids.begin(),dirty_ids.end(),std::greater<int>());
        for(int id:dirty_ids){
            const auto& node=nodes[id];double terminal=node.position->outcome();
            if(terminal>=0)values[id]=terminal==.5?0.:-1.;
            else if(node.children.empty())values[id]=-node.bootstrap;
            else{
                double hi=-std::numeric_limits<double>::infinity(),mass=0.,psum=0.;
                for(const auto& e:node.children)hi=std::max(hi,e.child<0?-node.bootstrap:-values[e.child]);
                for(const auto& e:node.children){
                    double q=e.child<0?-node.bootstrap:-values[e.child];
                    mass+=e.prior*std::exp((q-hi)/.1);psum+=e.prior;
                }
                values[id]=hi+.1*std::log(mass/psum);
            }
            dirty[id]=0;
        }
        dirty_ids.clear();
    }
    bool stage_done()const{
        for(const auto& b:branches)if(b.next<(int)b.pulls.size())return false;
        return true;
    }
    void plan_stage(){
        if(!pending.empty())throw std::runtime_error("stage before pending update");
        refresh();branches.clear();
        for(int owner=0;owner<(int)roots.size();++owner){
            int root=roots[owner],offset=branches.size(),current=nodes[root].n;
            auto& edges=nodes[root].children;
            std::vector<double> tilted(edges.size()),weights(edges.size());
            std::vector<int> counts(edges.size());
            double hi=-std::numeric_limits<double>::infinity(),total=0.;
            for(int j=0;j<(int)edges.size();++j){
                const auto& e=edges[j];double q=e.child<0?-nodes[root].bootstrap:-values[e.child];
                tilted[j]=std::log(e.prior)+beta[owner]*q;hi=std::max(hi,tilted[j]);
            }
            for(double& x:tilted){x=std::exp(x-hi);total+=x;}
            for(int j=0;j<(int)edges.size();++j){
                const auto& e=edges[j];double p=e.prior,pi=tilted[j]/total;
                weights[j]=std::sqrt(std::max(0.,(1-mixture)*p*(1-p)+mixture*pi*(1-pi)));
                counts[j]=e.child<0?0:nodes[e.child].n;
                branches.push_back(ForestBranch{owner,j,0,{}});
            }
            int stop=std::min(budgets[owner],current+block);
            for(int pull=current+1;pull<=stop;++pull){
                int best=0;double best_u=-std::numeric_limits<double>::infinity();
                for(int j=0;j<(int)edges.size();++j){
                    double u=weights[j]/(1+counts[j]);
                    if(u>best_u){best_u=u;best=j;}
                }
                branches[offset+best].pulls.push_back(pull);counts[best]++;
            }
        }
        stages++;
    }
    py::array_t<int> next_adaptive(){
        if(!pending.empty())throw std::runtime_error("pending update");
        if(stage_done() && !done())plan_stage();
        int before=nodes.size();auto out=next_handles();
        for(int id=before;id<(int)nodes.size();++id)mark(id);
        return out;
    }
    py::list planned()const{
        py::list out;
        for(const auto& b:branches)out.append(py::make_tuple(b.owner,b.edge,b.pulls));
        return out;
    }
};
PYBIND11_MODULE(_allie_adaptive_root,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<AdaptiveRootTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int,double,std::vector<double>,int>())
        .def("select",&AdaptiveRootTree::next_adaptive).def("update",&AdaptiveRootTree::update_fast)
        .def("snapshot",&AdaptiveRootTree::snapshot).def("backups",&AdaptiveRootTree::backups)
        .def("compact",&AdaptiveRootTree::compact).def("stats",&AdaptiveRootTree::stats)
        .def("prefix_evals",&AdaptiveRootTree::prefix_evals)
        .def_property_readonly("evals",[](const AdaptiveRootTree& x){return x.evals;})
        .def_property_readonly("done",&AdaptiveRootTree::done)
        .def_property_readonly("stage_done",&AdaptiveRootTree::stage_done)
        .def_property_readonly("stages",[](const AdaptiveRootTree& x){return x.stages;})
        .def_property_readonly("planned",&AdaptiveRootTree::planned)
        .def_property_readonly("values",[](AdaptiveRootTree& x){x.refresh();return x.values;});
}
