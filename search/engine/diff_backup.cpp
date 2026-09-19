#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
namespace py=pybind11;
using IA=py::array_t<int,py::array::c_style|py::array::forcecast>;
using DA=py::array_t<double,py::array::c_style|py::array::forcecast>;
struct Backup {
 std::vector<int> first,next,roots,move,degree;
 std::vector<double> prior,base,mass,term,lc;
 int n;
 Backup(py::dict d,int budget){
  IA dd=d["degree"].cast<IA>();
  IA parent=d["parent"].cast<IA>(),born=d["born"].cast<IA>(),rr=d["roots"].cast<IA>(),mm=d["move"].cast<IA>();
  DA pp=d["prior"].cast<DA>(),bb=d["boot"].cast<DA>(),ss=d["mass"].cast<DA>(),tt=d["terminal"].cast<DA>();
  n=parent.size();first.assign(n,-1);next.assign(n,-1);
  degree.assign(dd.data(),dd.data()+n);
  roots.assign(rr.data(),rr.data()+rr.size());move.assign(mm.data(),mm.data()+mm.size());
  prior.assign(pp.data(),pp.data()+n);base.resize(n);mass.assign(ss.data(),ss.data()+n);term.assign(tt.data(),tt.data()+n);lc.resize(n);
  std::vector<int> desc(n,1);
  for(int i=0;i<n;++i){base[i]=-bb.data()[i];int p=parent.data()[i];
   if(p>=i)throw std::invalid_argument("parent order");
   if(p>=0&&born.data()[i]<=budget){next[i]=first[p];first[p]=i;}
  }
  for(int i=n-1;i>=0;--i){for(int c=first[i];c>=0;c=next[c])desc[i]+=desc[c];lc[i]=std::log1p((desc[i]-1.)/16.);}
 }
 py::array_t<double> reduce(double a,double b,IA ids){
  int r=roots.size(),k=ids.shape(1);if(ids.ndim()!=2||ids.shape(0)!=r)throw std::invalid_argument("root IDs shape");
  py::array_t<double> out({3,r,k});std::fill(out.mutable_data(),out.mutable_data()+3*r*k,0.);
  std::vector<double> v(n),ga(n,0.),gb(n,0.);
  for(int i=n-1;i>=0;--i){
   if(term[i]>=0){v[i]=term[i]==.5?0.:-1.;continue;}
   if(mass[i]==0||first[i]<0){v[i]=base[i];continue;}
   int active=0;
   double tau=std::exp(a+b*lc[i]),seen=0.,hi=-std::numeric_limits<double>::infinity();
   for(int c=first[i];c>=0;c=next[c]){active++;seen+=prior[c];if(prior[c]>0)hi=std::max(hi,-v[c]);}
   double rest=active==degree[i]?0.:std::max(0.,mass[i]-seen);
   if(rest>0)hi=std::max(hi,base[i]);
   double tail=rest>0?rest*std::exp((base[i]-hi)/tau):0.;
   double z=tail,mean=tail*base[i],da=0.,db=0.;
   for(int c=first[i];c>=0;c=next[c]){
    double w=prior[c]*std::exp((-v[c]-hi)/tau);z+=w;mean-=w*v[c];da-=w*ga[c];db-=w*gb[c];
   }
   v[i]=hi+tau*std::log(z/mass[i]);double local=v[i]-mean/z;
   ga[i]=local+da/z;gb[i]=lc[i]*local+db/z;
  }
  for(int row=0;row<r;++row){
   int root=roots[row];std::vector<int> child(1968,-1);
   for(int c=first[root];c>=0;c=next[c])child[move[c]]=c;
   for(int j=0;j<k;++j){int id=ids.data(row,j)[0];if(id<0||id>=1968)throw std::invalid_argument("move ID");
    int c=child[id];out.mutable_at(0,row,j)=c<0?base[root]:-v[c];
    if(c>=0){out.mutable_at(1,row,j)=-ga[c];out.mutable_at(2,row,j)=-gb[c];}
   }
  }
  return out;
 }
};
PYBIND11_MODULE(_allie_diff_backup,m){py::class_<Backup>(m,"Backup").def(py::init<py::dict,int>()).def("reduce",&Backup::reduce);}
