// Chess rules (native/chess-library, MIT, pinned in README.md) and the search trees.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <tuple>
#include <unordered_map>
#include "chess.hpp"
namespace py=pybind11;
static std::vector<std::string> moves;
static std::unordered_map<std::string,int> ids;

struct Position {
    chess::Board board;
    Position()=default;
    explicit Position(const std::string& fen):board(fen){}
    void push(int token) {
        if (token<378 || token>=378+(int)moves.size()) throw std::invalid_argument("not a move token");
        auto m=chess::uci::uciToMove(board,moves[token-378]);
        if (!board.isLegal(m)) throw std::invalid_argument("illegal move");
        board.makeMove(m);
    }
    Position child(int token) const {Position p=*this;p.push(token);return p;}
    std::vector<int> legal() const {
        chess::Movelist ml;chess::movegen::legalmoves(ml,board);
        std::vector<std::pair<std::array<int,4>,int>> sorted;
        const bool check=board.inCheck();
        for(auto m:ml){
            auto u=chess::uci::moveToUci(m);
            int from=(u[1]-'1')*8+u[0]-'a',to=(u[3]-'1')*8+u[2]-'a';
            auto piece=board.at(m.from()).type();int cat=0,a=-from,b=-to,promo=0;
            if(m.typeOf()==chess::Move::CASTLING){cat=1;a=-int(m.to().index());}
            else if(piece==chess::PieceType::KING && check)cat=-1;
            else if(piece==chess::PieceType::PAWN){
                if(m.typeOf()==chess::Move::ENPASSANT)cat=5;
                else if(from%8!=to%8)cat=2;
                else {cat=std::abs(from-to)==16?4:3;a=-to;b=0;}
                if(u.size()==5)promo=u[4]=='q'?0:u[4]=='r'?1:u[4]=='b'?2:3;
            }
            sorted.push_back({{cat,a,b,promo},ids.at(u)});
        }
        std::sort(sorted.begin(),sorted.end());std::vector<int> result;
        for(auto &x:sorted)result.push_back(x.second);
        return result;
    }
    // -1: ongoing, otherwise expected score for White. Match python-chess
    // outcome(claim_draw=False): checkmate, insufficient, stalemate, 75m, 5fold.
    double outcome() const {
        chess::Movelist ml;chess::movegen::legalmoves(ml,board);
        if(ml.empty()) return board.inCheck()?(board.sideToMove()==chess::Color::WHITE?0.:1.):.5;
        auto heavy=board.pieces(chess::PieceType::PAWN)|board.pieces(chess::PieceType::ROOK)|board.pieces(chess::PieceType::QUEEN);
        if(!heavy){
            auto bishops=board.pieces(chess::PieceType::BISHOP),knights=board.pieces(chess::PieceType::KNIGHT);
            constexpr uint64_t dark=0xAA55AA55AA55AA55ULL;
            if(!bishops && knights.count()<=1)return .5;
            uint64_t bb=bishops.getBits();
            if(!knights && (!(bb&dark)||!(bb&~dark)))return .5;
        }
        if(board.halfMoveClock()>=150 || board.isRepetition(4))return .5;
        return -1.;
    }
};


// The original Allie's MCTS. Its output policy is computed in Python (policy.output).
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

// Per-root evaluation counts, and tree serialization for the value backups (value.cpp).
struct CompactTree : NativeMCTS {
    std::vector<int64_t> evals;
    std::vector<int> born;
    CompactTree(const std::vector<std::vector<int>>& prefixes, Scores scores,
                std::vector<int> sims, std::vector<double> cpuct)
        : NativeMCTS(prefixes,scores,std::move(sims),std::move(cpuct)), evals(roots.size(),0), born(nodes.size(),0) {}
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

// Exact parallel scheduling of independent root-action subtrees.
// Quotas depend on root priors only; preserve each branch's logical simulation
// order and node birth time so prefix-budget reductions remain well defined.

struct ForestBranch {
    int owner,edge,next=0;
    std::vector<int> pulls;
};
struct HandleForestTree : CompactTree {
    std::vector<ForestBranch> branches;
    int remaining=0,rounds=0;
    HandleForestTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
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
    void next_forest(){
        if(!pending.empty())throw std::runtime_error("update pending predictions first");
        if(done())throw std::runtime_error("already finished");
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
            else{pending.push_back(id);prefix_tokens+=nodes[id].prefix.size();evals[owner]++;}
            remaining--;
        }
        rounds++;iteration=done()?limit:rounds;
        if(!pending.empty())requests++;
        evaluated+=pending.size();
    }
    py::array_t<int> next_handles(){
        next_forest();
        py::array_t<int> out({int(pending.size()),4});
        for(int i=0;i<(int)pending.size();++i){
            int id=pending[i];int* row=out.mutable_data(i,0);
            row[0]=id;row[1]=nodes[id].parent;row[2]=nodes[id].move;row[3]=nodes[id].prefix.size();
        }
        return out;
    }
};


// Expand independent leaves concurrently; preserve exact serial backup order.
#include <exception>
struct ThreadForestTree : HandleForestTree {
    int threads;
    ThreadForestTree(const std::vector<std::vector<int>>& prefixes,Scores scores,
                     std::vector<int> sims,std::vector<double> cpuct,int nthreads)
        :HandleForestTree(prefixes,scores,std::move(sims),std::move(cpuct)),threads(nthreads){
        if(threads<1 || threads>8)throw std::invalid_argument("threads must be 1..8");
    }
    void update_fast(Scores scores){
        if(scores.ndim()!=2 || scores.shape(0)!=(int)pending.size() || scores.shape(1)!=2432)
            throw std::invalid_argument("leaf dimensions");
        int count=pending.size();const float* z=scores.data();
        std::vector<double> values(count);std::exception_ptr error;
        #pragma omp parallel for num_threads(threads) if(count>=128 && threads>1) schedule(static)
        for(int i=0;i<count;++i){
            try {values[i]=expand(pending[i],z+2432*i);}
            catch(...){
                #pragma omp critical
                {if(!error)error=std::current_exception();}
            }
        }
        if(error)std::rethrow_exception(error);
        for(int i=0;i<count;++i)backup(pending[i],values[i]);
        pending.clear();
    }
};

// Continue a completed root-coverage forest without re-evaluating any node.
struct GrowForest : ThreadForestTree {
    using ThreadForestTree::ThreadForestTree;
    void grow(const std::vector<int>& next_budgets) {
        if(!done()) throw std::invalid_argument("finish current phase before grow");
        if(next_budgets.size()!=roots.size()) throw std::invalid_argument("budget shape");
        for(int i=0;i<(int)roots.size();++i)
            if(next_budgets[i]<budgets[i]) throw std::invalid_argument("cannot shrink");
        int offset=0;
        for(int owner=0;owner<(int)roots.size();++owner) {
            int root=roots[owner],size=nodes[root].children.size();
            std::vector<double> weights(size);
            std::vector<int> counts(size,0);
            for(int j=0;j<size;++j) {
                double p=nodes[root].children[j].prior;
                weights[j]=std::sqrt(p*(1-p));
                if(branches[offset+j].next!=(int)branches[offset+j].pulls.size())
                    throw std::runtime_error("unfinished branch");
            }
            for(int pull=1;pull<=next_budgets[owner];++pull) {
                int best=0;double best_u=-std::numeric_limits<double>::infinity();
                for(int j=0;j<size;++j) {
                    double u=weights[j]/(1+counts[j]);
                    if(u>best_u){best_u=u;best=j;}
                }
                auto& branch=branches[offset+best];
                if(pull<=budgets[owner]) {
                    if(counts[best]>=(int)branch.pulls.size() || branch.pulls[counts[best]]!=pull)
                        throw std::runtime_error("quota prefix changed");
                } else {
                    branch.pulls.push_back(pull);remaining++;
                }
                counts[best]++;
            }
            offset+=size;
        }
        budgets=next_budgets;limit=*std::max_element(budgets.begin(),budgets.end());
    }
};

// NativeMCTS, queried through node handles.
struct HandleMCTS: NativeMCTS {
    using NativeMCTS::NativeMCTS;
    py::array_t<int> handles(){
        select();py::array_t<int> out({int(pending.size()),4});
        for(int i=0;i<int(pending.size());++i){int id=pending[i];int* p=out.mutable_data(i,0);
            p[0]=id;p[1]=nodes[id].parent;p[2]=nodes[id].move;p[3]=nodes[id].prefix.size();}
        return out;
    }
};

PYBIND11_MODULE(_allie_search_tree,m) {
 m.def("initialize",[](const std::vector<std::string>& v){
  if(v.size()!=1968)throw std::invalid_argument("vocabulary");
  moves=v;ids.clear();for(int i=0;i<int(v.size());++i)ids[v[i]]=378+i;});
 py::class_<Position>(m,"Position",py::module_local())
  .def(py::init<>()).def(py::init<const std::string&>())
  .def("push",&Position::push).def("child",&Position::child).def("legal",&Position::legal)
  .def("outcome",&Position::outcome).def("fen",[](const Position&p){return p.board.getFen();})
  .def_property_readonly("white",[](const Position&p){return p.board.sideToMove()==chess::Color::WHITE;});
 py::class_<GrowForest>(m,"Coverage",py::module_local())
  .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>,int>())
  .def("select",&GrowForest::next_handles).def("update",&GrowForest::update_fast)
  .def("grow",&GrowForest::grow).def("compact",&GrowForest::compact).def("stats",&GrowForest::stats)
  .def_property_readonly("evals",[](const GrowForest& x){return x.evals;})
  .def_property_readonly("done",&GrowForest::done);
 py::class_<HandleMCTS>(m,"Allie",py::module_local())
  .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
  .def_readwrite("first_prior",&HandleMCTS::first_prior).def_readwrite("preserve_depth",&HandleMCTS::preserve_depth)
  .def("select",&HandleMCTS::handles).def("update",&HandleMCTS::update)
  .def("summaries",&HandleMCTS::summaries).def("stats",&HandleMCTS::stats)
  .def_property_readonly("done",[](const HandleMCTS& x){return x.iteration>=x.limit && x.pending.empty();});
}
