// Released Allie tree mechanics. Python retains the exact FP32 output solver.
#include <cmath>
#include <memory>
#include <limits>
#include <pybind11/numpy.h>

struct SearchEdge {
    int move,child=-1;
    double prior;
    SearchEdge(int m,double p):move(m),prior(p){}
};

struct SearchNode {
    double prior=0.,w=0.,bootstrap=0.;
    int parent=-1,move=-1,depth=0,n=0;
    std::vector<SearchEdge> children;
    std::vector<int> prefix;
    std::unique_ptr<Position> position;
};

struct NativeMCTS {
    using Scores=py::array_t<float,py::array::c_style|py::array::forcecast>;
    std::vector<SearchNode> nodes;
    std::vector<int> roots,budgets,pending;
    std::vector<double> cp;
    int iteration=0,limit=0,max_depth=0,requests=0;
    bool first_prior=false,preserve_depth=false;
    int max_search_depth=100;
    int64_t depth_visits=0;
    int64_t evaluated=0,terminal_visits=0,prefix_tokens=0;

    int add(int parent,int move,double prior){
        SearchNode x;x.parent=parent;x.move=move;x.prior=prior;
        x.depth=parent<0?0:nodes[parent].depth+1;
        int id=nodes.size();nodes.push_back(std::move(x));return id;
    }

    void materialize(int id){
        if(nodes[id].position)return;
        int parent=nodes[id].parent;
        nodes[id].position=std::make_unique<Position>(nodes[parent].position->child(nodes[id].move));
        nodes[id].prefix=nodes[parent].prefix;nodes[id].prefix.push_back(nodes[id].move);
    }

    double expand(int id,const float* z){
        auto legal=nodes[id].position->legal();
        if(legal.empty())throw std::runtime_error("terminal expansion");
        double max=-std::numeric_limits<double>::infinity(),sum=0.;
        for(int t:legal)max=std::max(max,double(z[t]));
        std::vector<double> p;for(int t:legal){p.push_back(std::exp(double(z[t])-max));sum+=p.back();}
        nodes[id].children.clear();nodes[id].children.reserve(legal.size());
        for(size_t j=0;j<legal.size();++j)nodes[id].children.emplace_back(legal[j],p[j]/sum);
        max=std::max({double(z[2413]),double(z[2414]),double(z[2415])});
        double win=std::exp(double(z[2413])-max),draw=std::exp(double(z[2414])-max),loss=std::exp(double(z[2415])-max);
        nodes[id].bootstrap=loss/(win+draw+loss)-win/(win+draw+loss);
        return nodes[id].bootstrap;
    }

    NativeMCTS(const std::vector<std::vector<int>>& prefixes,Scores scores,
               std::vector<int> sims,std::vector<double> cpuct):budgets(std::move(sims)),cp(std::move(cpuct)){
        if(scores.ndim()!=2 || scores.shape(0)!=(int)prefixes.size() || scores.shape(1)!=2432 || budgets.size()!=prefixes.size() || cp.size()!=prefixes.size())
            throw std::invalid_argument("root dimensions");
        nodes.reserve(prefixes.size()*128);
        for(size_t i=0;i<prefixes.size();++i){
            int id=add(-1,-1,0.);roots.push_back(id);nodes[id].prefix=prefixes[i];
            nodes[id].position=std::make_unique<Position>();
            for(size_t j=11;j<prefixes[i].size();++j)nodes[id].position->push(prefixes[i][j]);
            if(nodes[id].position->outcome()>=0)throw std::invalid_argument("terminal root");
            if(prefixes[i].size()>=1025 || budgets[i]<0)throw std::invalid_argument("no search context or negative budget");
            expand(id,scores.data(i,0));limit=std::max(limit,budgets[i]);
        }
    }

    void backup(int id,double value){
        while(id>=0){nodes[id].n++;nodes[id].w+=value;value=-value;id=nodes[id].parent;}
    }

    std::vector<std::vector<int>> select(){
        if(!pending.empty())throw std::runtime_error("update pending predictions first");
        if(iteration>=limit)throw std::runtime_error("already finished");
        std::vector<std::vector<int>> prefixes;
        for(size_t i=0;i<roots.size();++i){
            if(budgets[i]<=iteration)continue;
            int id=roots[i],depth_limit=std::min(max_search_depth,1025-int(nodes[id].prefix.size()));
            while(!nodes[id].children.empty() && nodes[id].depth<depth_limit){
                double factor=(std::log((nodes[id].n+19652.+1)/19652.)+cp[i])*std::sqrt(double(first_prior?std::max(nodes[id].n,1):nodes[id].n));
                int best=-1;double best_u=-std::numeric_limits<double>::infinity();
                for(size_t j=0;j<nodes[id].children.size();++j){
                    auto& edge=nodes[id].children[j];
                    int n=edge.child<0?0:nodes[edge.child].n;
                    double q=n?nodes[edge.child].w/n:0.;
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
            if(outcome>=0){backup(id,outcome==.5?0.:1.);terminal_visits++;}
            else if(preserve_depth && !nodes[id].children.empty() && nodes[id].depth>=depth_limit){backup(id,nodes[id].bootstrap);depth_visits++;}
            else{pending.push_back(id);prefixes.push_back(nodes[id].prefix);prefix_tokens+=nodes[id].prefix.size();}
        }
        iteration++;if(!pending.empty())requests++;
        evaluated+=pending.size();return prefixes;
    }

    void update(Scores scores){
        if(scores.ndim()!=2 || scores.shape(0)!=(int)pending.size() || scores.shape(1)!=2432)throw std::invalid_argument("leaf dimensions");
        for(size_t i=0;i<pending.size();++i){int id=pending[i];double value=expand(id,scores.data(i,0));backup(id,value);}
        pending.clear();
    }

    py::list summaries()const{
        if(iteration!=limit || !pending.empty())throw std::runtime_error("unfinished trees");
        py::list result;
        for(size_t i=0;i<roots.size();++i){
            auto& root=nodes[roots[i]];std::vector<int> ids,counts;std::vector<double> q,p;
            int total=0;for(auto& edge:root.children){
                int n=edge.child<0?0:nodes[edge.child].n;
                ids.push_back(edge.move-378);counts.push_back(n);
                q.push_back(n?nodes[edge.child].w/n:0.);p.push_back(edge.prior);total+=n;
            }
            if(total!=budgets[i] || root.n!=budgets[i])throw std::runtime_error("visit accounting");
            result.append(py::make_tuple(ids,counts,q,p));
        }
        return result;
    }

    py::dict stats()const{
        py::dict d;int64_t sims=0;for(int n:budgets)sims+=n;
        d["simulations"]=sims;d["evaluated_leaves"]=evaluated;d["terminal_visits"]=terminal_visits;
        d["useful_prefix_tokens"]=prefix_tokens;d["requests"]=requests;d["max_depth"]=max_depth;if(depth_visits)d["depth_limited_visits"]=depth_visits;return d;
    }
};
