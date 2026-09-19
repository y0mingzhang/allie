// Separate player/opponent selectivity on an unchanged, budget-truncated tree.
#include "diff_backup.cpp"
struct PerspectiveBackup : Backup {
 std::vector<int> parity;
 PerspectiveBackup(py::dict d,int budget):Backup(d,budget){
  IA parent=d["parent"].cast<IA>();parity.assign(n,0);
  for(int i=0;i<n;++i)if(parent.at(i)>=0)parity[i]=1-parity[parent.at(i)];
 }
 py::array_t<double> perspective(double own,double opponent,IA ids){
  if(!(own>0)||!(opponent>0))throw std::invalid_argument("positive temperature multiplier required");
  int r=roots.size();if(ids.ndim()!=2||ids.shape(0)!=r)throw std::invalid_argument("root IDs shape");int k=ids.shape(1);
  std::vector<double> v(n);
  for(int i=n-1;i>=0;--i){
   if(term[i]>=0){v[i]=term[i]==.5?0.:-1.;continue;}
   if(mass[i]==0||first[i]<0){v[i]=base[i];continue;}
   double multiplier=parity[i]?opponent:own;
   double tau=std::exp(std::log(.2)-.5*lc[i])*multiplier;
   double seen=0.,hi=-std::numeric_limits<double>::infinity();int active=0;
   for(int c=first[i];c>=0;c=next[c]){++active;seen+=prior[c];if(prior[c]>0)hi=std::max(hi,-v[c]);}
   double rest=active==degree[i]?0.:std::max(0.,mass[i]-seen);
   if(std::isinf(tau)){
    double mean=rest*base[i];
    for(int c=first[i];c>=0;c=next[c])mean-=prior[c]*v[c];
    v[i]=mean/mass[i];continue;
   }
   if(rest>0)hi=std::max(hi,base[i]);
   double z=rest>0?rest*std::exp((base[i]-hi)/tau):0.;
   for(int c=first[i];c>=0;c=next[c])if(prior[c]>0)z+=prior[c]*std::exp((-v[c]-hi)/tau);
   v[i]=hi+tau*std::log(z/mass[i]);
  }
  py::array_t<double> out({r,k});
  for(int row=0;row<r;++row){int root=roots[row];std::vector<int> child(1968,-1);
   for(int c=first[root];c>=0;c=next[c])child[move[c]]=c;
   for(int j=0;j<k;++j){int id=ids.at(row,j);if(id<0||id>=1968)throw std::invalid_argument("move ID");int c=child[id];out.mutable_at(row,j)=c<0?base[root]:-v[c];}
  }
  return out;
 }
};
PYBIND11_MODULE(_allie_perspective_backup,m){py::class_<PerspectiveBackup>(m,"Backup").def(py::init<py::dict,int>()).def("reduce",&PerspectiveBackup::perspective);}
