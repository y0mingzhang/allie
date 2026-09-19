// Attribution ablations; the previously frozen full projection is unchanged.
#include "bellman_projection.cpp"

struct Modes : Projection {
 std::vector<int> is_root;
 Modes(py::dict data,int budget):Projection(data,budget),is_root(n,0){
  for(int root:roots)is_root[root]=1;
 }
 py::dict project_mode(double strength,int mode){
  if(mode<0||mode>3)throw std::invalid_argument("mode0 full,1 root-only,2 no-root,3 upward-only");
  if(mode==0)return Projection::project(strength);
  if(strength<0||!std::isfinite(strength))throw std::invalid_argument("strength");
  std::vector<double> mean=original,var(n,1.),total(n,0.),residual(n,0.),posterior;
  std::vector<int> use(n,0);
  for(int i=0;i<n;++i)use[i]=active[i]&&term[i]<0&&first[i]>=0&&mass[i]>0&&
      (mode==1?is_root[i]:(mode==2?!is_root[i]:true));
  for(int i=n-1;i>=0;--i){
   if(!active[i])continue;
   if(term[i]>=0){var[i]=0.;continue;}
   if(strength==0||!use[i])continue;
   double child_mean=0.,child_var=0.;
   for(int c=first[i];c>=0;c=next[c]){
    double p=prior[c]/mass[i];child_mean+=p*mean[c];child_var+=p*p*var[c];
   }
   total[i]=1./strength+rest[i]*rest[i]+child_var;
   residual[i]=child_mean-rest[i]*original[i];
   var[i]=total[i]/(1.+total[i]);
   mean[i]=var[i]*(original[i]-residual[i]/total[i]);
  }
  posterior=mean;
  if(strength>0&&mode!=3)for(int i=0;i<n;++i){
   if(!use[i])continue;
   double error=posterior[i]+residual[i];
   for(int c=first[i];c>=0;c=next[c])
    posterior[c]=mean[c]-var[c]*(prior[c]/mass[i])*error/total[i];
  }
  py::array_t<double> out(n);int clipped=0,count=0;
  for(int i=0;i<n;++i){
   out.mutable_data()[i]=posterior[i];base[i]=strength==0?original[i]:std::clamp(posterior[i],-1.,1.);
   if(active[i]){count++;clipped+=std::abs(posterior[i])>1.;}
  }
  py::dict result;result["mean"]=out;result["clipped"]=clipped;result["active"]=count;return result;
 }
};
PYBIND11_MODULE(_allie_bellman_modes,m){
 py::class_<Modes>(m,"Modes").def(py::init<py::dict,int>()).def("project",&Modes::project_mode).def("reduce",&Modes::reduce);
}
