// Causal two-stage quota allocation on independent root-action trace prefixes.
// No target moves or future value observations enter the allocation.
#include "diff_backup.cpp"
#include <pybind11/stl.h>
struct Replay {
 std::vector<int> parent,born,roots,move,owner,branch,terminal_count;
 std::vector<double> terminal;
 std::vector<std::vector<int>> order;
 std::vector<std::vector<double>> prob;
 std::vector<std::vector<std::vector<int>>> pulls;
 int n,r;
 Replay(py::dict data,DA logits,const std::vector<std::vector<int>>& legal,int maximum){
  IA pa=data["parent"].cast<IA>(),bb=data["born"].cast<IA>(),rr=data["roots"].cast<IA>(),mm=data["move"].cast<IA>();
  DA tt=data["terminal"].cast<DA>();n=pa.size();r=rr.size();
  parent.assign(pa.data(),pa.data()+n);born.assign(bb.data(),bb.data()+n);roots.assign(rr.data(),rr.data()+r);
  move.assign(mm.data(),mm.data()+n);terminal.assign(tt.data(),tt.data()+n);
  if(logits.ndim()!=2||logits.shape(0)!=r||logits.shape(1)!=2432||legal.size()!=r)throw std::invalid_argument("root shape");
  order=legal;prob.resize(r);pulls.resize(r);owner.assign(n,-1);branch.assign(n,-1);
  for(int o=0;o<r;++o){
   owner[roots[o]]=o;int k=order[o].size();if(k<1)throw std::invalid_argument("empty legal list");
   double hi=-INFINITY,z=0.;for(int t:order[o])hi=std::max(hi,logits.at(o,t+378));
   for(int t:order[o]){prob[o].push_back(std::exp(logits.at(o,t+378)-hi));z+=prob[o].back();}
   for(double& p:prob[o])p/=z;
   pulls[o].resize(k);std::vector<int> count(k,0);
   for(int b=1;b<=maximum && k>1;++b){
    int best=0;double top=-INFINITY;
    for(int j=0;j<k;++j){double p=prob[o][j],u=std::sqrt(p*(1-p))/(1+count[j]);if(u>top){top=u;best=j;}}
    pulls[o][best].push_back(b);count[best]++;
   }
  }
  for(int i=0;i<n;++i){
   int p=parent[i];if(p<0)continue;if(p>=i)throw std::invalid_argument("parent order");owner[i]=owner[p];
   if(parent[p]<0){auto& ids=order[owner[i]];auto it=std::find(ids.begin(),ids.end(),move[i]);if(it==ids.end())throw std::runtime_error("move absent");branch[i]=it-ids.begin();}
   else branch[i]=branch[p];
   auto& trace=pulls[owner[i]][branch[i]];
   if(!std::binary_search(trace.begin(),trace.end(),born[i]))throw std::runtime_error("recorded birth not in reconstructed branch trace");
  }
 }
 py::dict cut(DA q128,DA q256,IA ids,int budget,int mode){
  if(budget<256||mode<0||mode>2||q128.ndim()!=2||q256.ndim()!=2||ids.ndim()!=2||ids.shape(0)!=r||q128.shape(0)!=r||q256.shape(0)!=r||q128.shape(1)!=ids.shape(1)||q256.shape(1)!=ids.shape(1))throw std::invalid_argument("settings");
  std::vector<std::vector<int>> cutoff(r);int capped=0;
  for(int o=0;o<r;++o){
   int k=order[o].size();cutoff[o].assign(k,0);if(k==1)continue;
   std::vector<int> count(k);std::vector<double> scale(k,1.),weight(k);double moment=0.;
   for(int j=0;j<k;++j){
    auto& trace=pulls[o][j];count[j]=std::upper_bound(trace.begin(),trace.end(),256)-trace.begin();
    int index=-1;for(int t=0;t<ids.shape(1);++t)if(ids.at(o,t)==order[o][j]){index=t;break;}
    if(index<0)throw std::runtime_error("legal move missing from Q columns");
    double d=q256.at(o,index)-q128.at(o,index);
    if(mode)scale[j]=std::sqrt(.03*.03+d*d)*(mode==2?std::sqrt(1.+count[j]):1.);
    moment+=prob[o][j]*scale[j]*scale[j];
   }
   for(int j=0;j<k;++j){double p=prob[o][j];weight[j]=std::sqrt(p*(1-p))*std::clamp(scale[j]/std::sqrt(moment),.25,4.);}
   for(int b=257;b<=budget;++b){
    int best=0;double top=-INFINITY;
    for(int j=0;j<k;++j){double u=weight[j]/(1+count[j]);if(u>top){top=u;best=j;}}
    // Refuse to score an arm if the trace cannot supply its chosen action.
    // Do not silently censor it or use future outcomes to pick a substitute.
    if(count[best]>=(int)pulls[o][best].size())throw std::runtime_error("counterfactual exceeds cached branch trace");
    count[best]++;
   }
   for(int j=0;j<k;++j)if(count[j])cutoff[o][j]=pulls[o][j][count[j]-1];
  }
  IA masked(n),evals(r);std::fill(evals.mutable_data(),evals.mutable_data()+r,0);
  for(int i=0;i<n;++i){
   bool active=parent[i]<0||born[i]<=cutoff[owner[i]][branch[i]];masked.mutable_data()[i]=active?0:1000000000;
   if(active&&parent[i]>=0&&terminal[i]<0)evals.mutable_data()[owner[i]]++;
   if(active&&parent[i]>=0&&masked.at(parent[i])!=0)throw std::runtime_error("disconnected retained trace");
  }
  py::dict out;out["born"]=masked;out["nodes"]=evals;return out;
 }
};
PYBIND11_MODULE(_allie_quota_replay,m){py::class_<Replay>(m,"Replay").def(py::init<py::dict,DA,const std::vector<std::vector<int>>&,int>()).def("cut",&Replay::cut);}
