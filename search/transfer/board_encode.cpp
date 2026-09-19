#include <cstdint>
#include <cstring>
#include <cstdlib>
extern "C" int encode_boards(const int64_t* tokens, int64_t rows, int64_t cols,
                             const int32_t* moves, uint8_t* output, int64_t* error) {
  const uint8_t start[64]={4,2,3,5,6,3,2,4,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    7,7,7,7,7,7,7,7,10,8,9,11,12,9,8,10};
  for(int64_t row=0;row<rows;++row){
    uint8_t board[64]={};int white=1,rights=0,ep=-1,active=0;
    for(int64_t col=0;col<cols;++col){
      int64_t at=row*cols+col;int t=tokens[at];
      auto fail=[&](int code){error[0]=row;error[1]=col;error[2]=t;return code;};
      if(col==0 && t!=2348)return fail(1);
      if(t<0 || t>=2350)return fail(2);
      if(t==2348){std::memcpy(board,start,64);white=1;rights=15;ep=-1;active=1;}
      else if(t>=378 && t<2346){
        if(!active)return fail(3);
        int from=moves[3*t],to=moves[3*t+1],promotion=moves[3*t+2];
        if(from<0 || from>=64 || to<0 || to>=64)return fail(4);
        int p=board[from],captured=board[to];
        if(!p || ((p<=6)!=bool(white)) || (captured && ((captured<=6)==bool(white))))return fail(5);
        int type=(p-1)%6+1;
        if(type==1 && to==ep && captured==0 && (from%8!=to%8)){
          int square=to+(white?-8:8);
          if(board[square]!=(white?7:1))return fail(6);
          board[square]=0;
        }
        board[from]=0;board[to]=promotion?(promotion+(white?0:6)):p;
        if(type==6){
          rights &= white?12:3;
          if(std::abs(to-from)==2){
            int rf=to>from?to+1:to-2,rt=(from+to)/2;
            if(board[rf]!=(white?4:10))return fail(7);
            board[rt]=board[rf];board[rf]=0;
          }
        }
        if(from==0 || to==0)rights&=~2;
        if(from==7 || to==7)rights&=~1;
        if(from==56 || to==56)rights&=~8;
        if(from==63 || to==63)rights&=~4;
        ep=(type==1 && std::abs(to-from)==16)?(to+from)/2:-1;
        white=1-white;
      } else if(t==2346 || t==2347){active=0;}
      uint8_t* dst=output+at*68;std::memcpy(dst,board,64);
      dst[64]=white;dst[65]=rights;dst[66]=ep<0?0:(ep%8+1);dst[67]=active;
    }
  }
  return 0;
}
