// Reverse-KL regularized values on fixed trees, not visit-count probabilities.
#include <pybind11/stl.h>
#include "diff_backup.cpp"
double reverse_value(const std::vector<double>& q,const std::vector<double>& weight,double lambda) {
    if(q.empty() || q.size()!=weight.size() || !(lambda>0) || !std::isfinite(lambda))
        throw std::invalid_argument("value inputs");
    double total=0.,maximum=-std::numeric_limits<double>::infinity();
    for(size_t j=0;j<q.size();++j) {
        if(!std::isfinite(q[j]) || weight[j]<0 || !std::isfinite(weight[j]))throw std::invalid_argument("q/prior");
        if(weight[j]>0){total+=weight[j];maximum=std::max(maximum,q[j]);}
    }
    if(!(total>0))throw std::invalid_argument("zero prior");
    std::vector<double> p(q.size()),gap(q.size());double lower=0.;
    for(size_t j=0;j<q.size();++j) {
        p[j]=weight[j]/total;gap[j]=(maximum-q[j])/lambda;
        if(p[j]>0)lower=std::max(lower,p[j]-gap[j]);
    }
    // t=(nu-max q)/lambda. Log-space bracket resolves tiny best-action priors.
    double lo=std::log(lower),hi=0.;
    for(int k=0;k<60;++k) {
        double mid=(lo+hi)/2,t=std::exp(mid),norm=0.;
        for(size_t j=0;j<q.size();++j)if(p[j]>0)norm+=p[j]/(t+gap[j]);
        if(norm>1)lo=mid;else hi=mid;
    }
    double t=std::exp((lo+hi)/2),penalty=0.;
    for(size_t j=0;j<q.size();++j)if(p[j]>0)penalty+=p[j]*std::log(t+gap[j]);
    return maximum+lambda*(t-1.-penalty);
}
struct ReverseBackup : Backup {
    using Backup::Backup;
    py::array_t<double> reduce_reverse(double lambda,double exponent,IA ids,bool visited_only=false) {
        int r=roots.size(),k=ids.shape(1);
        if(ids.ndim()!=2 || ids.shape(0)!=r)throw std::invalid_argument("root IDs shape");
        std::vector<double> v(n),q,p;
        for(int i=n-1;i>=0;--i) {
            if(term[i]>=0){v[i]=term[i]==.5?0.:-1.;continue;}
            if(mass[i]==0||first[i]<0){v[i]=base[i];continue;}
            q.clear();p.clear();double seen=0.;int active=0;
            for(int c=first[i];c>=0;c=next[c]) {
                active++;seen+=prior[c];q.push_back(-v[c]);p.push_back(prior[c]);
            }
            double rest=active==degree[i]?0.:std::max(0.,mass[i]-seen);
            if(rest>0 && !visited_only){q.push_back(base[i]);p.push_back(rest);}
            v[i]=reverse_value(q,p,lambda*std::exp(exponent*lc[i]));
        }
        py::array_t<double> out({r,k});
        for(int row=0;row<r;++row) {
            int root=roots[row];std::vector<int> child(1968,-1);
            for(int c=first[root];c>=0;c=next[c])child[move[c]]=c;
            for(int j=0;j<k;++j) {
                int id=ids.data(row,j)[0];if(id<0||id>=1968)throw std::invalid_argument("move ID");
                int c=child[id];out.mutable_at(row,j)=c<0?base[root]:-v[c];
            }
        }
        return out;
    }
};
PYBIND11_MODULE(_allie_reverse_backup,m) {
    m.def("value",&reverse_value);
    py::class_<ReverseBackup>(m,"Backup")
        .def(py::init<py::dict,int>())
        .def("reduce",&ReverseBackup::reduce_reverse,py::arg("lambda"),py::arg("exponent"),py::arg("ids"),py::arg("visited_only")=false);
}
