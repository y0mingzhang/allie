// Alternative soft-backup temperatures, on identical cached search trees.
#include "compact.cpp"
py::array_t<double> adaptive_reduce(py::dict data,int budget,int mode,double base_tau,double scale) {
    if(mode<0 || mode>4 || base_tau<=0 || scale<=0 || !std::isfinite(base_tau) || !std::isfinite(scale))
        throw std::invalid_argument("temperature");
    IArray par=data["parent"].cast<IArray>(),depth=data["depth"].cast<IArray>();
    IArray born=data["born"].cast<IArray>(),roots=data["roots"].cast<IArray>();
    IArray move=data["move"].cast<IArray>();
    IArray degree=data["degree"].cast<IArray>();
    DArray prior=data["prior"].cast<DArray>(),boot=data["boot"].cast<DArray>();
    DArray mass=data["mass"].cast<DArray>(),term=data["terminal"].cast<DArray>();
    int n=par.size();std::vector<int> first(n,-1),next(n,-1);
    for(int i=0;i<n;++i){int p=par.data()[i];if(p>=0 && born.data()[i]<=budget){
        if(p>=i)throw std::invalid_argument("non-topological parent");
        next[i]=first[p];first[p]=i;
    }}
    std::vector<double> value(n);std::vector<int> descendants(n,1);
    for(int i=n-1;i>=0;--i){
        if(born.data()[i]>budget)continue;
        if(term.data()[i]>=0){value[i]=term.data()[i]==.5?0.:-1.;continue;}
        double base=-boot.data()[i],psum=mass.data()[i];
        if(psum==0){value[i]=base;continue;}
        double tau=base_tau,seen=0.,mean=0.,second=0.;
        double hi=-std::numeric_limits<double>::infinity();int active=0;
        for(int c=first[i];c>=0;c=next[c]){seen+=prior.data()[c];mean-=prior.data()[c]*value[c];second+=prior.data()[c]*value[c]*value[c];descendants[i]+=descendants[c];hi=std::max(hi,-value[c]);active++;}
        double rest=std::max(0.,psum-seen);
        if(active<degree.data()[i])hi=std::max(hi,base);
        if(mode==1)tau*=std::sqrt(double(std::max(depth.data()[i],1)));
        if(mode==2)tau/=std::sqrt(double(std::max(depth.data()[i],1)));
        if(mode==3)tau*=std::sqrt(scale/(scale+descendants[i]-1.));
        if(mode==4){
            double avg=(mean+rest*base)/psum;
            double variance=std::max(0.,(second+rest*base*base)/psum-avg*avg);
            tau=std::clamp(base_tau*std::sqrt(variance),.01,.25);
        }
        if(std::isinf(tau)){value[i]=(mean+rest*base)/psum;continue;}
        if(tau==0){value[i]=hi;continue;}
        double weighted=rest*std::exp((base-hi)/tau);
        for(int c=first[i];c>=0;c=next[c])weighted+=prior.data()[c]*std::exp((-value[c]-hi)/tau);
        value[i]=hi+tau*std::log(weighted/psum);
    }
    py::array_t<double> out({int(roots.size()),1968});
    for(int r=0;r<roots.size();++r){
        int id=roots.data()[r];double* row=out.mutable_data(r,0);
        std::fill(row,row+1968,-boot.data()[id]);
        for(int c=first[id];c>=0;c=next[c])row[move.data()[c]]=-value[c];
    }
    return out;
}

PYBIND11_MODULE(_allie_adaptive_temperature,m){m.def("reduce",&adaptive_reduce);}
