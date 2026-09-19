// Exact Gaussian factor-tree mean; then the existing count-soft backup.
#include "diff_backup.cpp"

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
PYBIND11_MODULE(_allie_bellman_projection,m){
 py::class_<Projection>(m,"Projection")
 .def(py::init<py::dict,int>()).def("project",&Projection::project).def("reduce",&Projection::reduce);
}
