// Human-policy Monte Carlo continuation; counter-based per-trajectory RNG.
#include "board.cpp"
#include <cstdint>
#include <numeric>
struct RolloutState {
    Position position;
    int root,action,replica,node,length,depth=0;
    double value=0.;
    bool ended=false;
    std::vector<int> legal;
    std::vector<double> prob;
};
struct HumanRollouts {
    using Scores=NativeMCTS::Scores;
    std::vector<RolloutState> states;
    std::vector<int> lengths,global_ids,evals,pending;
    std::vector<bool> root_white;
    int replicas,max_depth,depth=0,next_node;
    uint64_t seed;
    HumanRollouts(const std::vector<std::vector<int>>& prefixes,Scores logits,
                  std::vector<int> gids,int r,int md,uint64_t sd)
        :global_ids(std::move(gids)),replicas(r),max_depth(md),next_node(prefixes.size()),seed(sd){
        if(r<1||md<1||global_ids.size()!=prefixes.size()||logits.shape(0)!=(int)prefixes.size()||logits.shape(1)!=2432)
            throw std::invalid_argument("rollout dimensions");
        evals.resize(prefixes.size(),0);
        for(size_t i=0;i<prefixes.size();++i){
            Position p;
            for(size_t j=11;j<prefixes[i].size();++j)p.push(prefixes[i][j]);
            if(p.outcome()>=0||prefixes[i].size()>=1025)throw std::invalid_argument("invalid root");
            lengths.push_back(prefixes[i].size());root_white.push_back(p.board.sideToMove()==chess::Color::WHITE);
            auto legal=p.legal();
            for(int move:legal){
                RolloutState s;s.position=p;s.root=i;s.action=move;s.replica=-1;s.node=i;s.length=prefixes[i].size();
                s.ended=legal.size()==1;
                s.legal={move};s.prob={1.};states.push_back(std::move(s));
            }
        }
    }
    static uint64_t mix(uint64_t x){
        x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
    }
    double uniform(const RolloutState& s)const{
        uint64_t x=seed^mix(uint64_t(global_ids[s.root]))^mix(uint64_t(s.action)+98317)
            ^mix(uint64_t(s.replica+1)+81233)^mix(uint64_t(depth)+99871);
        return double(mix(x)>>11)*0x1.0p-53;
    }
    std::vector<std::array<int,4>> select(){
        if(!pending.empty()||depth>=max_depth)throw std::runtime_error("rollout select state");
        depth++;std::vector<std::array<int,4>> handles;
        for(size_t i=0;i<states.size();++i){
            auto& s=states[i];if(s.ended)continue;
            if(s.length>=1025){s.ended=true;continue;}
            int choice=0;
            if(depth>1){
                double u=uniform(s),sum=0.;choice=s.prob.size()-1;
                for(size_t j=0;j<s.prob.size();++j){sum+=s.prob[j];if(u<sum){choice=j;break;}}
            }
            int move=s.legal[choice],parent=s.node;s.position.push(move);s.length++;s.depth++;
            double outcome=s.position.outcome();
            if(outcome>=0){
                s.value=(2*outcome-1)*(root_white[s.root]?1:-1);s.ended=true;continue;
            }
            s.node=next_node++;pending.push_back(i);
            handles.push_back({s.node,parent,move,s.length});evals[s.root]++;
        }
        if(pending.empty()&&depth==1)fork();
        return handles;
    }
    void fork(){
        std::vector<RolloutState> all;all.reserve(states.size()*replicas);
        for(auto& s:states)for(int r=0;r<replicas;++r){auto copy=s;copy.replica=r;all.push_back(std::move(copy));}
        states=std::move(all);
    }
    void update(Scores z){
        if(z.ndim()!=2||z.shape(0)!=(int)pending.size()||z.shape(1)!=2432)throw std::invalid_argument("rollout scores");
        for(size_t i=0;i<pending.size();++i){
            auto& s=states[pending[i]];const float* row=z.data(i,0);
            double mx=std::max({double(row[2413]),double(row[2414]),double(row[2415])});
            double w=std::exp(row[2413]-mx),d=std::exp(row[2414]-mx),l=std::exp(row[2415]-mx);
            s.value=(w-l)/(w+d+l)*((s.position.board.sideToMove()==chess::Color::WHITE)==root_white[s.root]?1:-1);
            s.legal=s.position.legal();if(s.legal.empty())throw std::runtime_error("terminal queried");
            mx=-std::numeric_limits<double>::infinity();for(int t:s.legal)mx=std::max(mx,double(row[t]));
            double sum=0.;s.prob.clear();
            for(int t:s.legal){double p=std::exp(row[t]-mx);s.prob.push_back(p);sum+=p;}
            for(auto& p:s.prob)p/=sum;
        }
        pending.clear();if(depth==1)fork();
    }
    py::dict snapshot()const{
        if(!pending.empty())throw std::runtime_error("pending rollout values");
        py::array_t<double> q({int(lengths.size()),1968}),square({int(lengths.size()),1968});
        py::array_t<int> counts({int(lengths.size()),1968});
        std::fill(q.mutable_data(),q.mutable_data()+q.size(),0.);
        std::fill(square.mutable_data(),square.mutable_data()+square.size(),0.);
        std::fill(counts.mutable_data(),counts.mutable_data()+counts.size(),0);
        for(auto& s:states){
            *q.mutable_data(s.root,s.action-378)+=s.value;
            *square.mutable_data(s.root,s.action-378)+=s.value*s.value;
            (*counts.mutable_data(s.root,s.action-378))++;
        }
        for(int i=0;i<q.size();++i)if(counts.data()[i]){
            q.mutable_data()[i]/=counts.data()[i];square.mutable_data()[i]/=counts.data()[i];
            square.mutable_data()[i]=std::max(0.,square.data()[i]-q.data()[i]*q.data()[i]);
        }
        py::dict result;result["q"]=q;result["variance"]=square;result["counts"]=counts;return result;
    }
    py::list trace()const{
        py::list out;
        for(auto& s:states)out.append(py::make_tuple(s.root,s.action,s.replica,s.node,s.length,s.depth,s.value,s.ended,s.position.board.getFen()));
        return out;
    }
};
PYBIND11_MODULE(_allie_rollout,m){
    m.def("initialize",[](const std::vector<std::string>& vocab){moves=vocab;ids.clear();for(int i=0;i<(int)vocab.size();++i)ids[vocab[i]]=378+i;});
    py::class_<HumanRollouts>(m,"Rollouts",py::module_local())
      .def(py::init<const std::vector<std::vector<int>>&,HumanRollouts::Scores,std::vector<int>,int,int,uint64_t>())
      .def("select",&HumanRollouts::select).def("update",&HumanRollouts::update)
      .def("snapshot",&HumanRollouts::snapshot).def("trace",&HumanRollouts::trace)
      .def_readonly("evals",&HumanRollouts::evals).def_readonly("depth",&HumanRollouts::depth);
}
