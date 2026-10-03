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

// Corrected soft backup with a configurable count normalization.
struct ScaledCount : Backup {
 ScaledCount(py::dict d,int budget,double scale):Backup(d,budget){
  if(!(scale>0)||!std::isfinite(scale))throw std::invalid_argument("positive scale required");
  std::vector<int> count(n,1);
  for(int i=n-1;i>=0;--i){for(int c=first[i];c>=0;c=next[c])count[i]+=count[c];lc[i]=std::log1p((count[i]-1.)/scale);}
 }
};

// Exact Gaussian factor-tree mean; then the existing count-soft backup.

struct Projection : Backup {
 std::vector<double> original, rest;
 std::vector<int> active;
 Projection(py::dict data,int budget):Backup(data,budget),original(base),rest(n,0.),active(n,0){
  IA parent=data["parent"].cast<IA>(),born=data["born"].cast<IA>();
  for(int i=0;i<n;++i)active[i]=(parent.data()[i]<0||born.data()[i]<=budget);
  for(int i=0;i<n;++i){
   double sum=0.;int count=0;
   for(int c=first[i];c>=0;c=next[c]){sum+=prior[c];count++;}
   rest[i]=mass[i]>0 && count<degree[i]?std::max(0.,1.-sum/mass[i]):0.;
   if(term[i]>=0)original[i]=term[i]==.5?0.:-1.;
  }
 }
 py::dict project(double strength){
  if(strength<0||!std::isfinite(strength))throw std::invalid_argument("strength");
  std::vector<double> mean=original,var(n,1.),total(n,0.),residual(n,0.),posterior;
  for(int i=n-1;i>=0;--i){
   if(!active[i])continue;
   if(term[i]>=0){var[i]=0.;continue;}
   if(strength==0||first[i]<0||mass[i]<=0)continue;
   double child_mean=0.,child_var=0.;
   for(int c=first[i];c>=0;c=next[c]){
    double p=prior[c]/mass[i];child_mean+=p*mean[c];child_var+=p*p*var[c];
   }
   // Unseen aggregate value has prior mean y_i and variance1.
   total[i]=1./strength+rest[i]*rest[i]+child_var;
   residual[i]=child_mean-rest[i]*original[i];
   var[i]=total[i]/(1.+total[i]);
   mean[i]=var[i]*(original[i]-residual[i]/total[i]);
  }
  posterior=mean;
  if(strength>0)for(int i=0;i<n;++i){
   if(!active[i]||first[i]<0||term[i]>=0||mass[i]<=0)continue;
   double error=posterior[i]+residual[i];
   for(int c=first[i];c>=0;c=next[c])
    posterior[c]=mean[c]-var[c]*(prior[c]/mass[i])*error/total[i];
  }
  py::array_t<double> out(n),coverage(n);int clipped=0,count=0;
  for(int i=0;i<n;++i){
   out.mutable_data()[i]=posterior[i];coverage.mutable_data()[i]=1.-rest[i];
   base[i]=strength==0?original[i]:std::clamp(posterior[i],-1.,1.);
   if(active[i]){count++;clipped+=std::abs(posterior[i])>1.;}
  }
  py::dict result;result["mean"]=out;result["coverage"]=coverage;
  result["clipped"]=clipped;result["active"]=count;return result;
 }
};

PYBIND11_MODULE(_allie_search_value,m) {
 py::class_<ScaledCount>(m,"Backup").def(py::init<py::dict,int,double>()).def("reduce",&ScaledCount::reduce);
 py::class_<Projection>(m,"Projection").def(py::init<py::dict,int>()).def("project",&Projection::project).def("reduce",&Projection::reduce);
}
