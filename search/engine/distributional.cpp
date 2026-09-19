// Preserve the identical MCTS tree. Reconstruct full W/D/L visit sums from
// node visit counts and the stored categorical leaf predictions.
#include <array>
#include "backups.cpp"

struct DistributionalTree : BackupTree {
    using WDL=std::array<double,3>;
    std::unordered_map<int,WDL> predicted;
    static WDL probs(const float* z) {
        double hi=std::max({double(z[2413]),double(z[2414]),double(z[2415])});
        WDL a{std::exp(z[2413]-hi),std::exp(z[2414]-hi),std::exp(z[2415]-hi)};
        double s=a[0]+a[1]+a[2];for(double& x:a)x/=s;return a;
    }
    DistributionalTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                      std::vector<int> sims,std::vector<double> cpuct)
        : BackupTree(prefixes,scores,std::move(sims),std::move(cpuct)) {
        for(size_t i=0;i<roots.size();++i)predicted[roots[i]]=probs(scores.data(i,0));
    }
    void update_wdl(Scores scores) {
        if(scores.ndim()!=2 || scores.shape(0)!=(int)pending.size() || scores.shape(1)!=2432)
            throw std::invalid_argument("leaf dimensions");
        for(size_t i=0;i<pending.size();++i)predicted[pending[i]]=probs(scores.data(i,0));
        update(scores);
    }
    py::list distribution()const {
        if(!pending.empty())throw std::runtime_error("pending update");
        std::vector<WDL> sum(nodes.size());
        for(int id=int(nodes.size())-1;id>=0;--id){
            const auto& node=nodes[id];int local=node.n;
            for(const auto& edge:node.children)if(edge.child>=0){
                local-=nodes[edge.child].n;
                for(int j=0;j<3;++j)sum[id][j]+=sum[edge.child][2-j];
            }
            if(local<0)throw std::runtime_error("negative local visits");
            WDL boot{};double outcome=node.position->outcome();
            if(outcome>=0)boot=outcome==.5?WDL{0.,1.,0.}:WDL{0.,0.,1.};
            else boot=predicted.at(id);
            for(int j=0;j<3;++j)sum[id][j]+=local*boot[j];
            if(std::abs(sum[id][0]+sum[id][1]+sum[id][2]-node.n)>1e-8)
                throw std::runtime_error("WDL mass != visits");
            if(std::abs(sum[id][2]-sum[id][0]-node.w)>1e-8)
                throw std::runtime_error("WDL mean != original MCTS Q");
        }
        py::list result;
        for(int id:roots){
            std::vector<int> ids;std::vector<WDL> distributions;
            for(const auto& edge:nodes[id].children){
                ids.push_back(edge.move-378);
                // Unvisited actions retain Q=0; the draw placeholder is marked
                // by zero visits in the accompanying snapshot, not a prediction.
                WDL value{0.,1.,0.};
                if(edge.child>=0 && nodes[edge.child].n){
                    for(int j=0;j<3;++j)value[j]=sum[edge.child][2-j]/nodes[edge.child].n;
                }
                distributions.push_back(value);
            }
            result.append(py::make_tuple(ids,distributions));
        }
        return result;
    }
};

PYBIND11_MODULE(_allie_distributional,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    py::class_<DistributionalTree>(m,"Tree",py::module_local())
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def("select",&DistributionalTree::next).def("update",&DistributionalTree::update_wdl)
        .def("snapshot",&DistributionalTree::snapshot).def("distribution",&DistributionalTree::distribution)
        .def("backups",&DistributionalTree::backups)
        .def("stats",&DistributionalTree::stats)
        .def_property_readonly("evals",[](const DistributionalTree& x){return x.evals;});
}
