// Chess rules only. See vendor/chess-library/LICENSE (MIT), pinned in README.md.
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

#include "mcts_native.hpp"

PYBIND11_MODULE(_allie_board,m){
    m.def("initialize",[](const std::vector<std::string>& vocabulary){
        if(vocabulary.size()!=1968)throw std::invalid_argument("wrong vocabulary");
        moves=vocabulary;ids.clear();for(int i=0;i<(int)moves.size();++i)ids[moves[i]]=378+i;
    });
    py::class_<Position>(m,"Position")
        .def(py::init<>()).def(py::init<const std::string&>())
        .def("push",&Position::push).def("child",&Position::child)
        .def("legal",&Position::legal).def("outcome",&Position::outcome)
        .def("fen",[](const Position&p){return p.board.getFen();})
        .def_property_readonly("white",[](const Position&p){return p.board.sideToMove()==chess::Color::WHITE;});
    py::class_<NativeMCTS>(m,"NativeMCTS")
        .def(py::init<const std::vector<std::vector<int>>&,NativeMCTS::Scores,std::vector<int>,std::vector<double>>())
        .def("select",&NativeMCTS::select).def("update",&NativeMCTS::update)
        .def("summaries",&NativeMCTS::summaries).def("stats",&NativeMCTS::stats)
        .def_property_readonly("done",[](const NativeMCTS& x){return x.iteration>=x.limit && x.pending.empty();});
}
