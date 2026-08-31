#include "transpose_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::ostringstream s; s << "CUDA error at " << __FILE__ << ':' << __LINE__ \
    << ": " << cudaGetErrorString(e) << " (" << static_cast<int>(e) << ')'; \
  throw std::runtime_error(s.str()); } } while (0)

namespace {
constexpr std::uint32_t kSeed = 0x03C0FFEEu;
constexpr std::uint32_t kGuardBits = 0x7FC12345u;
enum class Pattern { Coordinate, Sequential, Signed };

const char* patternName(Pattern p) {
  switch (p) { case Pattern::Coordinate: return "coordinate";
    case Pattern::Sequential: return "sequential"; case Pattern::Signed: return "signed_seeded"; }
  return "unknown";
}

std::vector<float> makeInput(std::size_t width, std::size_t height, Pattern p) {
  std::vector<float> v(width * height); std::mt19937 rng(kSeed);
  std::uniform_int_distribution<std::int32_t> d(-1000000, 1000000);
  for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x) {
    const std::size_t i = y * width + x;
    if (p == Pattern::Coordinate) v[i] = static_cast<float>((y % 10000) * 10000 + (x % 10000));
    else if (p == Pattern::Sequential) v[i] = static_cast<float>(i);
    else v[i] = static_cast<float>(d(rng)) / 1000.0f;
  }
  return v;
}
void cpuTranspose(const std::vector<float>& in, std::vector<float>& out,
                  std::size_t width, std::size_t height) {
  for (std::size_t y = 0; y < height; ++y) for (std::size_t x = 0; x < width; ++x)
    out[x * height + y] = in[y * width + x];
}
struct Guarded { float* raw = nullptr; float* logical = nullptr; std::size_t count = 0; static constexpr std::size_t kGuard = 16;
  void allocate(std::size_t n) { count=n; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), (n + 2*kGuard)*sizeof(float))); logical=raw+kGuard; CUDA_CHECK(cudaMemset(raw, 0, (n+2*kGuard)*sizeof(float))); }
  void fillGuard() { std::vector<std::uint32_t> h(count + 2*kGuard, kGuardBits); CUDA_CHECK(cudaMemcpy(raw, h.data(), h.size()*sizeof(float), cudaMemcpyHostToDevice)); }
  bool check() const { std::vector<std::uint32_t> h(2*kGuard); CUDA_CHECK(cudaMemcpy(h.data(), raw, kGuard*sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h.data()+kGuard, raw+(kGuard+count), kGuard*sizeof(float), cudaMemcpyDeviceToHost)); return std::all_of(h.begin(),h.end(),[](auto x){return x==kGuardBits;}); }
  ~Guarded(){ if(raw) cudaFree(raw); }
};
struct Row { std::string version, pattern; std::size_t width,height; int repeat; int gridX,gridY,blockX,blockY; bool pass,guard; std::string cudaStatus,error; };
std::string utcTimestamp() { std::time_t t=std::time(nullptr); std::tm tm{}; gmtime_r(&t,&tm); std::ostringstream s; s<<std::put_time(&tm,"%Y-%m-%dT%H:%M:%SZ"); return s.str(); }

Row runCase(TransposeVersion version, Pattern pattern, std::size_t width, std::size_t height, int repeat) {
  const bool tiled = version == TransposeVersion::V3Tiled || version == TransposeVersion::V4Padded;
  const int gridY = static_cast<int>((height + (tiled ? TILE_DIM : BLOCK_ROWS) - 1) / (tiled ? TILE_DIM : BLOCK_ROWS));
  Row row{versionName(version),patternName(pattern),width,height,repeat,static_cast<int>((width+TILE_DIM-1)/TILE_DIM),gridY,TILE_DIM,BLOCK_ROWS,false,false,"PASS",""};
  try {
    const auto input=makeInput(width,height,pattern); std::vector<float> expected(width*height), actual(width*height);
    if (version == TransposeVersion::V1Copy) expected=input; else cpuTranspose(input,expected,width,height);
    float *dIn=nullptr; Guarded dOut; CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIn),input.size()*sizeof(float))); dOut.allocate(actual.size()); dOut.fillGuard();
    CUDA_CHECK(cudaMemcpy(dIn,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    launchTranspose(version,dIn,dOut.logical,width,height); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(),dOut.logical,actual.size()*sizeof(float),cudaMemcpyDeviceToHost)); row.guard=dOut.check();
    row.pass=row.guard && std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float))==0; if(!row.pass && row.error.empty()) row.error="bitwise mismatch or guard corruption";
    CUDA_CHECK(cudaFree(dIn)); row.cudaStatus="PASS";
  } catch(const std::exception& e) { row.cudaStatus="FAIL"; row.error=e.what(); }
  return row;
}
void writeHeader(std::ostream& o){o<<"timestamp,version,width,height,input_pattern,repeat,grid_x,grid_y,block_x,block_y,tile_dim,block_rows,correctness,guard_status,cuda_status,error_message\n";}
void writeRow(std::ostream& o,const Row&r){o<<utcTimestamp()<<','<<r.version<<','<<r.width<<','<<r.height<<','<<r.pattern<<','<<r.repeat<<','<<r.gridX<<','<<r.gridY<<','<<r.blockX<<','<<r.blockY<<','<<TILE_DIM<<','<<BLOCK_ROWS<<','<<(r.pass?"PASS":"FAIL")<<','<<(r.guard?"PASS":"FAIL")<<','<<r.cudaStatus<<",\""<<r.error<<"\"\n";}
}

__global__ void copyKernel(const float* in,float* out,std::size_t w,std::size_t h){std::size_t x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x<w&&y<h)out[y*w+x]=in[y*w+x];}
__global__ void naiveKernel(const float* in,float* out,std::size_t w,std::size_t h){std::size_t x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x<w&&y<h)out[x*h+y]=in[y*w+x];}
template<int PAD> __global__ void tiledKernel(const float* in,float* out,std::size_t w,std::size_t h){__shared__ float tile[TILE_DIM][TILE_DIM+PAD];std::size_t x=blockIdx.x*TILE_DIM+threadIdx.x,y=blockIdx.y*TILE_DIM+threadIdx.y;for(int j=0;j<TILE_DIM;j+=BLOCK_ROWS){std::size_t yy=y+j;if(x<w&&yy<h)tile[threadIdx.y+j][threadIdx.x]=in[yy*w+x];}__syncthreads();std::size_t ox=blockIdx.y*TILE_DIM+threadIdx.x,oy=blockIdx.x*TILE_DIM+threadIdx.y;for(int j=0;j<TILE_DIM;j+=BLOCK_ROWS){std::size_t yy=oy+j;if(ox<h&&yy<w)out[yy*h+ox]=tile[threadIdx.x][threadIdx.y+j];}}

const char* versionName(TransposeVersion v){switch(v){case TransposeVersion::V1Copy:return"V1";case TransposeVersion::V2Naive:return"V2";case TransposeVersion::V3Tiled:return"V3";case TransposeVersion::V4Padded:return"V4";}return"unknown";}
void launchTranspose(TransposeVersion v,const float* in,float* out,std::size_t w,std::size_t h,void* stream){dim3 b(TILE_DIM,BLOCK_ROWS); const bool tiled = v == TransposeVersion::V3Tiled || v == TransposeVersion::V4Padded; dim3 g((w+TILE_DIM-1)/TILE_DIM,(h+(tiled?TILE_DIM:BLOCK_ROWS)-1)/(tiled?TILE_DIM:BLOCK_ROWS));cudaStream_t s=reinterpret_cast<cudaStream_t>(stream);switch(v){case TransposeVersion::V1Copy:copyKernel<<<g,b,0,s>>>(in,out,w,h);break;case TransposeVersion::V2Naive:naiveKernel<<<g,b,0,s>>>(in,out,w,h);break;case TransposeVersion::V3Tiled:tiledKernel<0><<<g,b,0,s>>>(in,out,w,h);break;case TransposeVersion::V4Padded:tiledKernel<1><<<g,b,0,s>>>(in,out,w,h);break;}}

int main(int argc,char**argv){try{std::string raw,summary;bool correctness=false;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--correctness")correctness=true;else if(a=="--raw"&&i+1<argc)raw=argv[++i];else if(a=="--summary"&&i+1<argc)summary=argv[++i];else if(a=="--device-info"){cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));std::cout<<p.name<<" cc "<<p.major<<'.'<<p.minor<<"\n";return 0;}}if(!correctness)throw std::invalid_argument("use --correctness --raw FILE --summary FILE");if(raw.empty()||summary.empty())throw std::invalid_argument("--raw and --summary are required");std::ofstream out(raw);writeHeader(out);std::vector<Row> rows;const std::pair<std::size_t,std::size_t> dims[]={{1,1},{17,1},{1,17},{7,13},{13,7},{31,31},{32,32},{33,33},{31,32},{32,31},{32,33},{33,32},{63,65},{65,63},{127,129},{129,127},{511,513},{513,511},{997,1000},{1000,997},{2048,2048},{4093,4096},{4096,4093}};const Pattern pats[]={Pattern::Coordinate,Pattern::Sequential,Pattern::Signed};for(auto d:dims)for(auto p:pats)for(int v=1;v<=4;++v)for(int r=1;r<=3;++r){auto row=runCase(static_cast<TransposeVersion>(v-1),p,d.first,d.second,r);writeRow(out,row);rows.push_back(row);}out.close();std::ofstream sum(summary);sum<<"version,total,pass,fail,guard_fail,cuda_fail\n";for(int v=1;v<=4;++v){int total=0,pass=0,gfail=0,cfail=0;for(auto&r:rows)if(r.version=="V"+std::to_string(v)){++total;pass+=r.pass;gfail+=!r.guard;cfail+=r.cudaStatus!="PASS";}sum<<"V"<<v<<','<<total<<','<<pass<<','<<total-pass<<','<<gfail<<','<<cfail<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
