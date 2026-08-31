#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iterator>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::ostringstream s; s << "CUDA error at " << __FILE__ << ':' << __LINE__ \
    << ": " << cudaGetErrorString(e); throw std::runtime_error(s.str()); } } while (0)

namespace {
constexpr std::uint32_t kGuard = 0x7FC12345u;
constexpr float kAtol = 1.0e-3f;
constexpr float kRtol = 1.0e-4f;
struct Shape { std::size_t m, k, n; };

std::string timestamp() { std::time_t t=std::time(nullptr); std::tm tm{}; gmtime_r(&t,&tm); std::ostringstream s; s<<std::put_time(&tm,"%Y-%m-%dT%H:%M:%SZ"); return s.str(); }

std::vector<float> makeInput(std::size_t count, std::uint32_t seed) {
  std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(count); for (float& x : v) x = d(rng); return v;
}

void cpuGemm(const std::vector<float>& a, const std::vector<float>& b,
             std::vector<float>& c, Shape s) {
  std::fill(c.begin(), c.end(), 0.0f);
  for (std::size_t row=0; row<s.m; ++row)
    for (std::size_t k=0; k<s.k; ++k) {
      const float av = a[row*s.k+k];
      for (std::size_t col=0; col<s.n; ++col) c[row*s.n+col] += av*b[k*s.n+col];
    }
}

__global__ void gemmV1(const float* a, const float* b, float* c,
                       std::size_t m, std::size_t k, std::size_t n) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.y)*blockDim.y + threadIdx.y;
  const std::size_t col = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float acc = 0.0f;
  for (std::size_t i=0; i<k; ++i) acc += a[row*k+i] * b[i*n+col];
  c[row*n+col] = acc;
}

void launchV1(const float* a,const float* b,float* c,Shape s,cudaStream_t stream=nullptr) {
  constexpr unsigned BX=16, BY=16;
  dim3 block(BX,BY); dim3 grid(static_cast<unsigned>((s.n+BX-1)/BX), static_cast<unsigned>((s.m+BY-1)/BY));
  gemmV1<<<grid,block,0,stream>>>(a,b,c,s.m,s.k,s.n); CUDA_CHECK(cudaPeekAtLastError());
}

struct Guarded { float* raw=nullptr; float* logical=nullptr; std::size_t count=0; static constexpr std::size_t width=16;
  void allocate(std::size_t n) { count=n; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw),(n+2*width)*sizeof(float))); logical=raw+width; }
  void fill() { std::vector<std::uint32_t> h(count+2*width,kGuard); CUDA_CHECK(cudaMemcpy(raw,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice)); }
  bool intact() const { std::vector<std::uint32_t> h(2*width); CUDA_CHECK(cudaMemcpy(h.data(),raw,width*sizeof(float),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h.data()+width,raw+(width+count),width*sizeof(float),cudaMemcpyDeviceToHost)); return std::all_of(h.begin(),h.end(),[](std::uint32_t x){return x==kGuard;}); }
  ~Guarded(){ if(raw) cudaFree(raw); }
};

struct ErrorStats { double maxAbs=0,maxRel=0,rmse=0,scale=0; bool pass=false,guard=true; };
ErrorStats compare(const std::vector<float>& ref,const std::vector<float>& got,bool guard) {
  ErrorStats e; e.guard=guard; double sum=0; for(std::size_t i=0;i<ref.size();++i){ double r=ref[i],g=got[i],a=std::abs(g-r); e.maxAbs=std::max(e.maxAbs,a); e.scale=std::max(e.scale,std::abs(r)); e.maxRel=std::max(e.maxRel,a/std::max(std::abs(r),1.0e-12)); sum+=a*a; } e.rmse=std::sqrt(sum/ref.size()); e.pass=e.guard && e.maxAbs <= kAtol + kRtol*e.scale; return e;
}

ErrorStats runCorrectnessCase(Shape s, std::ostream& out) {
  const auto a=makeInput(s.m*s.k,0x04020001u+static_cast<unsigned>(s.m)); const auto b=makeInput(s.k*s.n,0x04020002u+static_cast<unsigned>(s.n)); std::vector<float> ref(s.m*s.n),got(s.m*s.n); cpuGemm(a,b,ref,s);
  float *da=nullptr,*db=nullptr; Guarded dc; ErrorStats stats;
  try { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&db),b.size()*sizeof(float))); dc.allocate(got.size()); dc.fill(); CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); launchV1(da,db,dc.logical,s); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(got.data(),dc.logical,got.size()*sizeof(float),cudaMemcpyDeviceToHost)); stats=compare(ref,got,dc.intact()); }
  catch(const std::exception& e){ out<<timestamp()<<','<<s.m<<','<<s.k<<','<<s.n<<",V0,V1,ERROR,ERROR,ERROR,ERROR,FAIL,FAIL,\""<<e.what()<<"\"\n"; if(da) cudaFree(da); if(db) cudaFree(db); throw; }
  out<<timestamp()<<','<<s.m<<','<<s.k<<','<<s.n<<",FP32,V1,"<<std::setprecision(10)<<stats.maxAbs<<','<<stats.maxRel<<','<<stats.rmse<<','<<(kAtol+kRtol*stats.scale)<<','<<(stats.guard?"PASS":"FAIL")<<','<<(stats.pass?"PASS":"FAIL")<<",\"\"\n";
  if(da) CUDA_CHECK(cudaFree(da)); if(db) CUDA_CHECK(cudaFree(db)); return stats;
}

double eventMs(Shape s,const float* a,const float* b,float* c,int iterations) { cudaEvent_t st{},sp{}; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&sp)); CUDA_CHECK(cudaEventRecord(st)); for(int i=0;i<iterations;++i) launchV1(a,b,c,s); CUDA_CHECK(cudaEventRecord(sp)); CUDA_CHECK(cudaEventSynchronize(sp)); float total=0; CUDA_CHECK(cudaEventElapsedTime(&total,st,sp)); CUDA_CHECK(cudaEventDestroy(st)); CUDA_CHECK(cudaEventDestroy(sp)); return total/iterations; }

double calibrate(Shape s,const float* a,const float* b,float* c) { for(int i=0;i<20;++i) launchV1(a,b,c,s); CUDA_CHECK(cudaDeviceSynchronize()); return eventMs(s,a,b,c,20); }

void benchmark(Shape s,int trial,double calibrated,const std::string& path) {
  std::vector<float> a=makeInput(s.m*s.k,0x1234u),b=makeInput(s.k*s.n,0x5678u); float *da=nullptr,*db=nullptr,*dc=nullptr; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&db),b.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dc),s.m*s.n*sizeof(float))); CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice)); launchV1(da,db,dc,s); CUDA_CHECK(cudaDeviceSynchronize()); int warm=std::max(20,static_cast<int>(std::ceil(1000.0/std::max(calibrated,1e-6)))); warm=std::min(warm,20000); int reps=std::max(100,static_cast<int>(std::ceil(1000.0/std::max(calibrated,1e-6)))); reps=std::min(reps,10000); for(int i=0;i<warm;++i) launchV1(da,db,dc,s); CUDA_CHECK(cudaDeviceSynchronize()); cudaEvent_t st{},sp{}; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&sp)); CUDA_CHECK(cudaEventRecord(st)); for(int i=0;i<reps;++i) launchV1(da,db,dc,s); CUDA_CHECK(cudaEventRecord(sp)); CUDA_CHECK(cudaEventSynchronize(sp)); float total=0; CUDA_CHECK(cudaEventElapsedTime(&total,st,sp)); CUDA_CHECK(cudaEventDestroy(st)); CUDA_CHECK(cudaEventDestroy(sp)); double latency=total/reps; double gflops=(2.0*static_cast<double>(s.m)*s.n*s.k)/(latency*1e6); std::ofstream out(path,std::ios::app); if(!out) throw std::runtime_error("cannot open benchmark output"); out<<timestamp()<<",V1,"<<s.m<<','<<s.k<<','<<s.n<<','<<trial<<','<<warm<<','<<reps<<','<<std::setprecision(12)<<calibrated<<','<<total<<','<<latency<<','<<gflops<<'\n'; CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
}

void deviceInfo(){ cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0)); int rt=0; CUDA_CHECK(cudaRuntimeGetVersion(&rt)); std::cout<<"gpu_name="<<p.name<<"\ncompute_capability="<<p.major<<'.'<<p.minor<<"\ncuda_runtime_version="<<rt<<"\nsm_count="<<p.multiProcessorCount<<"\nwarp_size="<<p.warpSize<<"\nblock=16x16\n"; }
}

int main(int argc,char** argv){ try { std::string mode,raw; Shape s{}; int trial=1; double calibrated=0; for(int i=1;i<argc;++i){std::string a=argv[i]; auto val=[&](){if(++i>=argc)throw std::invalid_argument("missing option");return std::string(argv[i]);}; if(a=="--device-info")mode="info"; else if(a=="--correctness")mode="correctness"; else if(a=="--benchmark-single")mode="bench"; else if(a=="--calibrate")mode="calibrate"; else if(a=="--m")s.m=std::stoull(val()); else if(a=="--k")s.k=std::stoull(val()); else if(a=="--n")s.n=std::stoull(val()); else if(a=="--raw")raw=val(); else if(a=="--trial")trial=std::stoi(val()); else if(a=="--calibrated-ms")calibrated=std::stod(val()); else throw std::invalid_argument("unknown option "+a); }
  if(mode=="info"){deviceInfo();return 0;} if(mode=="correctness"){if(raw.empty())throw std::invalid_argument("--raw required"); std::ofstream out(raw); out<<"timestamp,M,K,N,precision,version,max_abs_error,max_rel_error,rmse,tolerance,guard_status,correctness,error_message\n"; const Shape cases[]={{1,1,1},{2,3,4},{15,16,17},{16,17,15},{17,15,16},{31,32,33},{32,33,31},{33,31,32},{128,256,64},{257,129,513},{63,65,97},{511,513,127},{512,512,512}}; int pass=0; for(Shape c:cases){auto e=runCorrectnessCase(c,out); pass+=e.pass;} std::cout<<"cases="<<std::size(cases)<<" pass="<<pass<<"\n"; return pass==static_cast<int>(std::size(cases))?0:2; }
  if(!s.m||!s.k||!s.n)throw std::invalid_argument("M,K,N required"); if(mode=="calibrate"){float *a=nullptr,*b=nullptr,*c=nullptr; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a),s.m*s.k*sizeof(float))); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b),s.k*s.n*sizeof(float))); CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c),s.m*s.n*sizeof(float))); double ms=calibrate(s,a,b,c); std::cout<<std::setprecision(12)<<ms<<"\n"; cudaFree(a);cudaFree(b);cudaFree(c);return 0;} if(mode=="bench"){if(raw.empty()||calibrated<=0)throw std::invalid_argument("benchmark requires --raw --calibrated-ms"); benchmark(s,trial,calibrated,raw);return 0;} throw std::invalid_argument("use --device-info, --correctness, --calibrate or --benchmark-single"); } catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;} }
