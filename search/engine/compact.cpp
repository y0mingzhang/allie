// Read-only, lossless tree serialization for CPU-only backup experiments.
#include "backups.cpp"

struct CompactTree : BackupTree {
    std::vector<int> born;
    CompactTree(const std::vector<std::vector<int>>& prefixes, Scores scores,
                std::vector<int> sims, std::vector<double> cpuct)
        : BackupTree(prefixes,scores,std::move(sims),std::move(cpuct)), born(nodes.size(),0) {}
    std::vector<std::vector<int>> next_compact() {
        auto result=next();born.resize(nodes.size(),iteration);return result;
    }
    py::dict compact()const {
        if(!pending.empty())throw std::runtime_error("pending update");
        size_t n=nodes.size();
        py::array_t<int> parent(n),move(n),depth(n),birth(n),degree(n),root(roots.size());
        py::array_t<double> prior(n),boot(n),mass(n),terminal(n);
        for(size_t i=0;i<n;++i){
            const auto& node=nodes[i];
            parent.mutable_data()[i]=node.parent;move.mutable_data()[i]=node.move-378;
            depth.mutable_data()[i]=node.depth;birth.mutable_data()[i]=born[i];
            degree.mutable_data()[i]=node.children.size();
            prior.mutable_data()[i]=node.prior;boot.mutable_data()[i]=node.bootstrap;
            double p=0.;for(const auto& e:node.children)p+=e.prior;
            mass.mutable_data()[i]=p;terminal.mutable_data()[i]=node.position->outcome();
        }
        std::copy(roots.begin(),roots.end(),root.mutable_data());
        py::dict out;
        out["parent"]=parent;out["move"]=move;out["depth"]=depth;out["born"]=birth;
        out["prior"]=prior;out["boot"]=boot;out["mass"]=mass;
        out["terminal"]=terminal;out["roots"]=root;out["degree"]=degree;
        return out;
    }
};

using DArray=py::array_t<double,py::array::c_style|py::array::forcecast>;
using IArray=py::array_t<int,py::array::c_style|py::array::forcecast>;

// Same soft backup as BackupTree, with independently chosen temperatures for
// root-player and opponent turns. Collapsed unseen edges all use own bootstrap.
py::array_t<double> reduce(py::dict data,int budget,double own_tau,double opp_tau) {
    if(own_tau<0 || opp_tau<0 || std::isnan(own_tau) || std::isnan(opp_tau))
        throw std::invalid_argument("temperature");
    IArray par=data["parent"].cast<IArray>(),depth=data["depth"].cast<IArray>();
    IArray born=data["born"].cast<IArray>(),roots=data["roots"].cast<IArray>();
    IArray move=data["move"].cast<IArray>();
    IArray degree=data["degree"].cast<IArray>();
    DArray prior=data["prior"].cast<DArray>(),boot=data["boot"].cast<DArray>();
    DArray mass=data["mass"].cast<DArray>(),term=data["terminal"].cast<DArray>();
    int n=par.size();std::vector<int> first(n,-1),next(n,-1);
    for(int i=0;i<n;++i){int p=par.data()[i];if(p>=0 && born.data()[i]<=budget){
        if(p>=i)throw std::invalid_argument("non-topological parent");
        next[i]=first[p];first[p]=i;
    }}
    std::vector<double> value(n);
    for(int i=n-1;i>=0;--i){
        if(born.data()[i]>budget)continue;
        if(term.data()[i]>=0){value[i]=term.data()[i]==.5?0.:-1.;continue;}
        double base=-boot.data()[i],psum=mass.data()[i];
        if(psum==0){value[i]=base;continue;}
        double tau=(depth.data()[i]%2?opp_tau:own_tau),seen=0.,mean=0.;
        double hi=-std::numeric_limits<double>::infinity();int active=0;
        for(int c=first[i];c>=0;c=next[c]){seen+=prior.data()[c];mean-=prior.data()[c]*value[c];hi=std::max(hi,-value[c]);active++;}
        double rest=std::max(0.,psum-seen);
        if(active<degree.data()[i])hi=std::max(hi,base);
        if(std::isinf(tau)){value[i]=(mean+rest*base)/psum;continue;}
        if(tau==0){value[i]=hi;continue;}
        double weighted=rest*std::exp((base-hi)/tau);
        for(int c=first[i];c>=0;c=next[c])weighted+=prior.data()[c]*std::exp((-value[c]-hi)/tau);
        value[i]=hi+tau*std::log(weighted/psum);
    }
    py::array_t<double> out({int(roots.size()),1968});
    for(int r=0;r<roots.size();++r){
        int id=roots.data()[r];double* row=out.mutable_data(r,0);
        std::fill(row,row+1968,-boot.data()[id]);
        for(int c=first[id];c>=0;c=next[c])row[move.data()[c]]=-value[c];
    }
    return out;
}

PYBIND11_MODULE(_allie_compact,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    m.def("reduce",&reduce);
    py::class_<CompactTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def("select",&CompactTree::next_compact).def("update",&CompactTree::update)
        .def("snapshot",&CompactTree::snapshot).def("backups",&CompactTree::backups)
        .def("compact",&CompactTree::compact).def("stats",&CompactTree::stats)
        .def_property_readonly("evals",[](const CompactTree& x){return x.evals;});
}
