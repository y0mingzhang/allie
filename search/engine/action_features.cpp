#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include "chess.hpp"
namespace py=pybind11;
py::array_t<double> encode(const std::vector<std::vector<int>>& prefixes,const std::vector<std::vector<int>>& legal,const std::vector<std::string>& vocab) {
 if(prefixes.size()!=legal.size() || vocab.size()!=1968)throw std::invalid_argument("inputs");
 int n=prefixes.size(),k=0;for(auto& v:legal)k=std::max(k,(int)v.size());
 py::array_t<double> out({n,k,20});std::fill(out.mutable_data(),out.mutable_data()+n*k*20,0.);
 for(int i=0;i<n;++i){
  chess::Board b;
  for(size_t j=11;j<prefixes[i].size();++j){int t=prefixes[i][j];if(t<378||t>=2346)throw std::invalid_argument("token");auto m=chess::uci::uciToMove(b,vocab[t-378]);if(!b.isLegal(m))throw std::invalid_argument("history");b.makeMove(m);}
  for(int j=0;j<(int)legal[i].size();++j){
   int t=legal[i][j];if(t<378||t>=2346)throw std::invalid_argument("candidate");auto m=chess::uci::uciToMove(b,vocab[t-378]);if(!b.isLegal(m))throw std::invalid_argument("candidate illegal");
   const auto& u=vocab[t-378];int ff=u[0]-'a',fr=u[1]-'1',tf=u[2]-'a',tr=u[3]-'1';
   int piece=int(b.at(m.from()).type());out.mutable_at(i,j,piece)=1.;
   if(b.isCapture(m)){int victim=m.typeOf()==chess::Move::ENPASSANT?0:int(b.at(m.to()).type());out.mutable_at(i,j,6+victim)=1.;}
   chess::Board c=b;c.makeMove(m);chess::Movelist replies;chess::movegen::legalmoves(replies,c);
   out.mutable_at(i,j,12)=c.inCheck();
   out.mutable_at(i,j,13)=m.typeOf()==chess::Move::CASTLING;
   out.mutable_at(i,j,14)=u.size()==5;
   out.mutable_at(i,j,15)=(b.sideToMove()==chess::Color::WHITE?tr:7-tr)/7.;
   out.mutable_at(i,j,16)=1-std::abs(tf-3.5)/3.5;
   out.mutable_at(i,j,17)=std::hypot(tf-ff,tr-fr)/std::sqrt(98.);
   out.mutable_at(i,j,18)=std::log1p(replies.size());
   out.mutable_at(i,j,19)=replies.size()==1;
  }
 }
 return out;
}
PYBIND11_MODULE(_allie_action_features,m){m.def("encode",&encode);}
