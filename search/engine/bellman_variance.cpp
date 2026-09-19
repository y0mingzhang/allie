// Heteroscedastic measurement regularization. Variances are heuristics, not CIs.
#include "bellman_projection.cpp"

struct Weighted : Projection {
 Weighted(py::dict data,int budget):Projection(data,budget){}
 py::dict weighted(double strength,DA variance){
  if(variance.ndim()!=1||variance.size()!=n)throw std::invalid_argument("variance shape");
  if(strength<0||!std::isfinite(strength))throw std::invalid_argument("strength");
  std::vector<double> measurement(variance.data(),variance.data()+n);
  for(double s:measurement)if(s<=0||!std::isfinite(s))throw std::invalid_argument("variance must be positive finite");
  bool uniform=std::all_of(measurement.begin(),measurement.end(),[](double s){return s==1.;});
  if(uniform||strength==0)return Projection::project(strength);
  std::vector<double> mean=original,var=measurement,total(n,0.),residual(n,0.),posterior;
  for(int i=n-1;i>=0;--i){
   if(!active[i])continue;
   if(term[i]>=0){var[i]=0.;continue;}
   if(first[i]<0||mass[i]<=0)continue;
   double child_mean=0.,child_var=0.;
   for(int c=first[i];c>=0;c=next[c]){
    double p=prior[c]/mass[i];child_mean+=p*mean[c];child_var+=p*p*var[c];
   }
   total[i]=1./strength+rest[i]*rest[i]*measurement[i]+child_var;
   residual[i]=child_mean-rest[i]*original[i];
   var[i]=measurement[i]*total[i]/(measurement[i]+total[i]);
   mean[i]=(total[i]*original[i]-measurement[i]*residual[i])/(measurement[i]+total[i]);
  }
  posterior=mean;
  for(int i=0;i<n;++i){
   if(!active[i]||first[i]<0||term[i]>=0||mass[i]<=0)continue;
   double error=posterior[i]+residual[i];
   for(int c=first[i];c>=0;c=next[c])
    posterior[c]=mean[c]-var[c]*(prior[c]/mass[i])*error/total[i];
  }
  py::array_t<double> out(n),coverage(n);int clipped=0,count=0;
  for(int i=0;i<n;++i){
   out.mutable_data()[i]=posterior[i];coverage.mutable_data()[i]=1.-rest[i];
   base[i]=std::clamp(posterior[i],-1.,1.);
   if(active[i]){count++;clipped+=std::abs(posterior[i])>1.;}
  }
  py::dict result;result["mean"]=out;result["coverage"]=coverage;
  result["clipped"]=clipped;result["active"]=count;return result;
 }
};
PYBIND11_MODULE(_allie_bellman_variance,m){
 py::class_<Weighted>(m,"Weighted").def(py::init<py::dict,int>()).def("project",&Weighted::weighted).def("reduce",&Weighted::reduce);
}
