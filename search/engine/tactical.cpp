// Bounded policy-guided quiescence. Rules only; every nonterminal value is neural.
#include <queue>
#include "board.cpp"
struct Tactical : NativeMCTS {
    using Candidate=std::tuple<double,int,int>;
    std::vector<std::priority_queue<Candidate>> queues;
    std::vector<int> owner,evals;
    std::vector<double> reach;
    int cap,depth_limit,branch;bool started=false;
    Tactical(const std::vector<std::vector<int>>& p,Scores z,int budget,int depth,int width)
        :NativeMCTS(p,z,std::vector<int>(p.size(),budget),std::vector<double>(p.size(),0)),cap(budget),depth_limit(depth),branch(width){
        if(cap<70||depth_limit<1||branch<1)throw std::invalid_argument("limits");
        queues.resize(p.size());evals.resize(p.size());owner.resize(nodes.size());reach.resize(nodes.size(),1.);
        for(size_t r=0;r<roots.size();r++)owner[roots[r]]=r;
    }
    void enqueue(int id){
        if(nodes[id].depth>=depth_limit||nodes[id].prefix.size()>=1025)return;
        const auto& b=nodes[id].position->board;bool check=b.inCheck();
        std::vector<std::pair<double,int>> eligible;
        for(size_t j=0;j<nodes[id].children.size();j++){
            const auto& e=nodes[id].children[j];auto m=chess::uci::uciToMove(b,moves[e.move-378]);
            if(check||b.isCapture(m)||m.typeOf()==chess::Move::PROMOTION)eligible.push_back({e.prior,int(j)});
        }
        std::stable_sort(eligible.begin(),eligible.end(),[](auto a,auto b){return a.first>b.first;});
        if(!check&&eligible.size()>size_t(branch))eligible.resize(branch);
        for(auto [p,j]:eligible)queues[owner[id]].push({reach[id]*p,id,j});
    }
    void take(int par,int edge,std::vector<std::vector<int>>& queries){
        if(nodes[par].children[edge].child>=0)throw std::runtime_error("duplicate edge");
        int r=owner[par],move=nodes[par].children[edge].move;double p=nodes[par].children[edge].prior;
        int id=add(par,move,p);owner.push_back(r);reach.push_back(nodes[par].depth==0?std::sqrt(p):reach[par]*p);
        nodes[par].children[edge].child=id;materialize(id);
        if(nodes[id].position->outcome()>=0)return;
        pending.push_back(id);queries.push_back(nodes[id].prefix);evals[r]++;
    }
    std::vector<std::vector<int>> select_round(){
        if(!pending.empty())throw std::runtime_error("pending");
        std::vector<std::vector<int>> queries;
        if(!started){
            for(int r:roots)for(size_t j=0;j<nodes[r].children.size();j++)take(r,int(j),queries);
            started=true;return queries;
        }
        for(size_t r=0;r<roots.size();r++){
            int used=0;
            while(evals[r]<cap&&!queues[r].empty()&&used<4){
                auto [score,id,j]=queues[r].top();queues[r].pop();int before=evals[r];take(id,j,queries);used+=evals[r]-before;
            }
        }
        return queries;
    }
    void update_tactical(Scores z){
        if(z.ndim()!=2||z.shape(0)!=(int)pending.size()||z.shape(1)!=2432)throw std::invalid_argument("scores");
        auto ps=pending;pending.clear();
        for(size_t j=0;j<ps.size();j++){expand(ps[j],z.data(j,0));enqueue(ps[j]);}
    }
    bool done()const{
        if(!started||!pending.empty())return false;
        for(size_t r=0;r<roots.size();r++)if(evals[r]<cap&&!queues[r].empty())return false;
        return true;
    }
    std::vector<double> values(double tau)const{
        if(tau<0||std::isnan(tau))throw std::invalid_argument("tau");
        std::vector<double> v(nodes.size());
        for(int i=int(nodes.size())-1;i>=0;i--){
            auto& node=nodes[i];double terminal=node.position->outcome();
            if(terminal>=0){v[i]=terminal==.5?0.:-1.;continue;}
            double base=-node.bootstrap;v[i]=base;
            if(node.children.empty())continue;
            double hi=-std::numeric_limits<double>::infinity(),mean=0.,sum=0.;
            for(const auto& e:node.children){double q=e.child<0?base:-v[e.child];hi=std::max(hi,q);mean+=e.prior*q;sum+=e.prior;}
            if(std::isinf(tau)){v[i]=mean/sum;continue;}
            if(tau==0){v[i]=hi;continue;}
            double ex=0.;for(const auto& e:node.children){double q=e.child<0?base:-v[e.child];ex+=e.prior*std::exp((q-hi)/tau);}
            v[i]=hi+tau*std::log(ex/sum);
        }
        return v;
    }
    py::array_t<double> q(double tau)const{
        auto v=values(tau);py::array_t<double> out({int(roots.size()),1968});std::fill(out.mutable_data(),out.mutable_data()+out.size(),0.);
        for(size_t r=0;r<roots.size();r++)for(auto& e:nodes[roots[r]].children)out.mutable_at(r,e.move-378)=e.child<0?-nodes[roots[r]].bootstrap:-v[e.child];
        return out;
    }
    py::list inspect()const{
        py::list result;
        for(auto& n:nodes){py::list edges;for(auto& e:n.children)edges.append(py::make_tuple(e.move,e.child,e.prior));result.append(py::make_tuple(n.parent,n.prefix,-n.bootstrap,n.position->outcome(),edges,n.depth));}
        return result;
    }
};
PYBIND11_MODULE(_allie_tactical,m){
    m.def("initialize",[](const std::vector<std::string>& v){moves=v;ids.clear();for(int i=0;i<(int)v.size();i++)ids[v[i]]=378+i;});
    py::class_<Tactical>(m,"Tree",py::module_local()).def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,int,int,int>())
    .def("select",&Tactical::select_round).def("update",&Tactical::update_tactical).def("q",&Tactical::q).def("inspect",&Tactical::inspect)
    .def_property_readonly("done",&Tactical::done).def_readonly("evals",&Tactical::evals).def_readonly("roots",&Tactical::roots);
}
