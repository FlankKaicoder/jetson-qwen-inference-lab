#include "transpose_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::ostringstream s;s<<cudaGetErrorString(e);throw std::runtime_error(s.str());} } while(0)

int parseVersion(const std::string& s){if(s.size()!=2||s[0]!='V'||s[1]<'1'||s[1]>'4')throw std::invalid_argument("version V1..V4 required");return s[1]-'1';}
void run(const std::string& vs,std::size_t w,std::size_t h,int warm,int reps,int trials,const std::string& outPath,bool single){
 const auto v=static_cast<TransposeVersion>(parseVersion(vs)); const std::size_t n=w*h; std::vector<float> host(n,1.0f),check(n); float *di=nullptr,*do_=nullptr; CUDA_CHECK(cudaMalloc((void**)&di,n*sizeof(float))); CUDA_CHECK(cudaMalloc((void**)&do_,n*sizeof(float))); CUDA_CHECK(cudaMemcpy(di,host.data(),n*sizeof(float),cudaMemcpyHostToDevice)); launchTranspose(v,di,do_,w,h); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(check.data(),do_,n*sizeof(float),cudaMemcpyDeviceToHost)); if(std::memcmp(host.data(),check.data(),n*sizeof(float))!=0)throw std::runtime_error("benchmark correctness sanity failed"); if(single){launchTranspose(v,di,do_,w,h);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());cudaFree(di);cudaFree(do_);return;} std::ofstream out(outPath,std::ios::app); if(!out)throw std::runtime_error("cannot open output"); for(int t=1;t<=trials;++t){for(int i=0;i<warm;++i){launchTranspose(v,di,do_,w,h);CUDA_CHECK(cudaGetLastError());}CUDA_CHECK(cudaDeviceSynchronize());cudaEvent_t a,b;CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));CUDA_CHECK(cudaEventRecord(a));for(int i=0;i<reps;++i){launchTranspose(v,di,do_,w,h);CUDA_CHECK(cudaGetLastError());}CUDA_CHECK(cudaEventRecord(b));CUDA_CHECK(cudaEventSynchronize(b));float total=0;CUDA_CHECK(cudaEventElapsedTime(&total,a,b));cudaEventDestroy(a);cudaEventDestroy(b);double ms=total/reps;double gb=(2.0*n*sizeof(float))/(ms*1e6);out<<vs<<','<<w<<','<<h<<','<<t<<','<<warm<<','<<reps<<','<<std::setprecision(10)<<ms<<','<<gb<<"\n";}cudaFree(di);cudaFree(do_);
}
int main(int argc,char**argv){try{std::string v,out;std::size_t w=0,h=0;int warm=20,reps=100,trials=5;bool single=false;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--version"&&i+1<argc)v=argv[++i];else if(a=="--width"&&i+1<argc)w=std::stoull(argv[++i]);else if(a=="--height"&&i+1<argc)h=std::stoull(argv[++i]);else if(a=="--warmup"&&i+1<argc)warm=std::stoi(argv[++i]);else if(a=="--repetitions"&&i+1<argc)reps=std::stoi(argv[++i]);else if(a=="--trials"&&i+1<argc)trials=std::stoi(argv[++i]);else if(a=="--output"&&i+1<argc)out=argv[++i];else if(a=="--single")single=true;else if(a=="--device-info"){cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));std::cout<<p.name<<" cc "<<p.major<<'.'<<p.minor<<"\n";return 0;}}if(v.empty()||!w||!h)throw std::invalid_argument("--version --width --height required");run(v,w,h,warm,reps,trials,out,single);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
