// Odd monotone utility in internal Bellman backups; returns expected-score scale.
#include "diff_backup.cpp"
struct GeometryBackup : Backup {
 using Backup::Backup;
 py::array_t<double> geometry(int kind,double strength,IA ids) {
  if(kind<0||kind>3)throw std::invalid_argument("kind");
  if((kind==1 && !(strength>0 && strength<1)) || (kind==2 && !(strength>0)))throw std::invalid_argument("strength");
  auto transform=[&](double x){x=std::clamp(x,-1.,1.);return kind==1?std::atanh(strength*x)/std::atanh(strength):kind==2?std::asinh(strength*x)/std::asinh(strength):x;};
  auto inverse=[&](double x){return kind==1?std::tanh(x*std::atanh(strength))/strength:kind==2?std::sinh(x*std::asinh(strength))/strength:x;};
  int r=roots.size();if(ids.ndim()!=2||ids.shape(0)!=r)throw std::invalid_argument("root IDs shape");int k=ids.shape(1);
  std::vector<double> v(n);
  for(int i=n-1;i>=0;--i){
   if(term[i]>=0){v[i]=term[i]==.5?0.:-1.;continue;}
   double b=transform(base[i]);
   if(mass[i]==0||first[i]<0){v[i]=b;continue;}
   double tau=.2*std::exp(-.5*lc[i]);
   if(kind==3)tau*=std::pow(.05+1-base[i]*base[i],strength);
   double seen=0.,hi=-std::numeric_limits<double>::infinity();int active=0;
   for(int c=first[i];c>=0;c=next[c]){++active;seen+=prior[c];if(prior[c]>0)hi=std::max(hi,-v[c]);}
   double rest=active==degree[i]?0.:std::max(0.,mass[i]-seen);
   if(rest>0)hi=std::max(hi,b);
   double z=rest>0?rest*std::exp((b-hi)/tau):0.;
   for(int c=first[i];c>=0;c=next[c])if(prior[c]>0)z+=prior[c]*std::exp((-v[c]-hi)/tau);
   v[i]=hi+tau*std::log(z/mass[i]);
  }
  py::array_t<double> out({r,k});
  for(int row=0;row<r;++row){int root=roots[row];std::vector<int> child(1968,-1);
   for(int c=first[root];c>=0;c=next[c])child[move[c]]=c;
   for(int j=0;j<k;++j){int id=ids.at(row,j);if(id<0||id>=1968)throw std::invalid_argument("move ID");int c=child[id];out.mutable_at(row,j)=c<0?base[root]:-inverse(v[c]);}
  }
  return out;
 }
};
PYBIND11_MODULE(_allie_geometry_backup,m){py::class_<GeometryBackup>(m,"Backup").def(py::init<py::dict,int>()).def("reduce",&GeometryBackup::geometry);}
