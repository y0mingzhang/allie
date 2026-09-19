// An isolated reducer; existing trees, selection and frozen binaries stay intact.
#include "compact.cpp"

py::array_t<double> discounted(py::dict data,int budget,double tau,double lambda) {
    if(tau<=0 || !std::isfinite(tau) || lambda<0 || lambda>1 || !std::isfinite(lambda))
        throw std::invalid_argument("discount parameters");
    IArray par=data["parent"].cast<IArray>(),born=data["born"].cast<IArray>();
    IArray roots=data["roots"].cast<IArray>(),move=data["move"].cast<IArray>();
    IArray degree=data["degree"].cast<IArray>();
    DArray prior=data["prior"].cast<DArray>(),boot=data["boot"].cast<DArray>();
    DArray mass=data["mass"].cast<DArray>(),term=data["terminal"].cast<DArray>();
    int n=par.size();std::vector<int> first(n,-1),next(n,-1);
    for(int i=0;i<n;++i){int p=par.data()[i];if(p>=0 && born.data()[i]<=budget){
        if(p>=i)throw std::invalid_argument("non-topological parent");
        next[i]=first[p];first[p]=i;
    }}
    std::vector<double> value(n);
    for(int i=n-1;i>=0;--i){
        if(born.data()[i]>budget)continue;
        if(term.data()[i]>=0){value[i]=term.data()[i]==.5?0.:-1.;continue;}
        double base=-boot.data()[i],psum=mass.data()[i];
        if(psum==0 || lambda==0){value[i]=base;continue;}
        double seen=0.,hi=-std::numeric_limits<double>::infinity();int active=0;
        for(int c=first[i];c>=0;c=next[c]){seen+=prior.data()[c];hi=std::max(hi,-value[c]);active++;}
        double rest=std::max(0.,psum-seen);
        if(active<degree.data()[i])hi=std::max(hi,base);
        double weighted=rest*std::exp((base-hi)/tau);
        for(int c=first[i];c>=0;c=next[c])weighted+=prior.data()[c]*std::exp((-value[c]-hi)/tau);
        double backed=hi+tau*std::log(weighted/psum);
        value[i]=(1-lambda)*base+lambda*backed;
    }
    py::array_t<double> out({int(roots.size()),1968});
    for(int r=0;r<roots.size();++r){
        int id=roots.data()[r];double* row=out.mutable_data(r,0);
        std::fill(row,row+1968,-boot.data()[id]);
        for(int c=first[id];c>=0;c=next[c])row[move.data()[c]]=-value[c];
    }
    return out;
}

PYBIND11_MODULE(_allie_discount,m){m.def("reduce",&discounted);}
