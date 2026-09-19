// Reuse the audited rules/tree implementation; the unused reference init function
// is not imported. This module adds read-only summaries and per-root accounting.
#define ALLIE_MODULE _allie_backup_unused_reference
#include "board.cpp"

struct BackupTree : NativeMCTS {
    std::vector<int64_t> evals;
    std::unordered_map<int,int> root_index;
    BackupTree(const std::vector<std::vector<int>>& prefixes, Scores scores,
               std::vector<int> sims, std::vector<double> cpuct)
        : NativeMCTS(prefixes,scores,std::move(sims),std::move(cpuct)), evals(roots.size(),0) {
        for(size_t i=0;i<roots.size();++i)root_index[roots[i]]=int(i);
        first_prior=preserve_depth=true;
    }
    std::vector<std::vector<int>> next() {
        auto result=select();
        for(int id:pending){while(nodes[id].parent>=0)id=nodes[id].parent;evals[root_index.at(id)]++;}
        return result;
    }
    py::list snapshot()const {
        if(!pending.empty())throw std::runtime_error("pending update");
        py::list result;
        for(int id:roots){
            std::vector<int> ids,counts;std::vector<double> q,p;
            int total=0;
            for(auto &edge:nodes[id].children){
                int n=edge.child<0?0:nodes[edge.child].n;
                ids.push_back(edge.move-378);counts.push_back(n);
                q.push_back(n?nodes[edge.child].w/n:0.);p.push_back(edge.prior);total+=n;
            }
            if(total!=nodes[id].n)throw std::runtime_error("visit accounting");
            result.append(py::make_tuple(ids,counts,q,p));
        }
        return result;
    }
    py::list backups(const std::vector<double>& temperatures)const {
        if(!pending.empty())throw std::runtime_error("pending update");
        py::list all;
        for(double tau:temperatures){
            if(tau<0 || std::isnan(tau))throw std::invalid_argument("temperature");
            // value is from the current mover's view; bootstrap is from their parent's view.
            std::vector<double> v(nodes.size());
            for(int id=int(nodes.size())-1;id>=0;--id){
                const auto& node=nodes[id];double terminal=node.position->outcome();
                if(terminal>=0){v[id]=terminal==.5?0.:-1.;continue;}
                double base=-node.bootstrap;
                if(node.children.empty()){v[id]=base;continue;}
                double hi=-std::numeric_limits<double>::infinity(),mean=0.,psum=0.;
                for(const auto& e:node.children){
                    double q=e.child<0?base:-v[e.child];
                    hi=std::max(hi,q);mean+=e.prior*q;psum+=e.prior;
                }
                if(std::isinf(tau)){v[id]=mean/psum;continue;}
                if(tau==0){v[id]=hi;continue;}
                double mass=0.;
                for(const auto& e:node.children){double q=e.child<0?base:-v[e.child];mass+=e.prior*std::exp((q-hi)/tau);}
                v[id]=hi+tau*std::log(mass/psum);
            }
            py::list roots_out;
            for(int id:roots){
                std::vector<int> ids;std::vector<double> q;
                for(const auto& e:nodes[id].children){ids.push_back(e.move-378);q.push_back(e.child<0?-nodes[id].bootstrap:-v[e.child]);}
                roots_out.append(py::make_tuple(ids,q));
            }
            all.append(roots_out);
        }
        return all;
    }
    py::list exported()const {
        if(!pending.empty())throw std::runtime_error("pending update");
        py::list result;
        for(const auto& node:nodes){
            py::list edges;
            for(const auto& e:node.children)edges.append(py::make_tuple(e.child,e.prior,e.move));
            result.append(py::make_tuple(node.parent,node.bootstrap,node.position->outcome(),edges));
        }
        return result;
    }
};

PYBIND11_MODULE(_allie_backups,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    py::class_<BackupTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def("select",&BackupTree::next).def("update",&BackupTree::update)
        .def("snapshot",&BackupTree::snapshot).def("backups",&BackupTree::backups)
        .def("exported",&BackupTree::exported).def("stats",&BackupTree::stats)
        .def_property_readonly("evals",[](const BackupTree& x){return x.evals;})
        .def_property_readonly("done",[](const BackupTree& x){return x.iteration>=x.limit && x.pending.empty();});
}
