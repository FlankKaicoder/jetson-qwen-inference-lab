#include <NvInfer.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTokens = 8;
constexpr int kHidden = 1024;
constexpr int kElements = kTokens * kHidden;
constexpr int kWarmup = 50;
constexpr int kRepetitions = 200;
constexpr int kTrials = 5;
constexpr float kEpsilon = 1.0e-6F;
constexpr char kPluginName[] = "RMSNormPlugin";
constexpr char kPluginVersion[] = "1";

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

template <typename T>
struct Destroy { void operator()(T* p) const noexcept { delete p; } };
template <typename T>
using TrtPtr = std::unique_ptr<T, Destroy<T>>;

void check(cudaError_t status, const char* op) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }

struct Buffer {
    void* ptr{nullptr};
    explicit Buffer(size_t bytes) { check(cudaMalloc(&ptr, bytes), "cudaMalloc"); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

std::vector<__half> readHalfFile(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "cannot open " + path);
    std::vector<__half> values(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(__half)));
    require(file.gcount() == static_cast<std::streamsize>(count * sizeof(__half)), "short input file " + path);
    return values;
}

struct Metrics { double relativeL2{}; double maxAbs{}; };
Metrics compare(const std::vector<__half>& a, const std::vector<__half>& b) {
    double an = 0.0, dn = 0.0, mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double av = __half2float(a[i]);
        const double bv = __half2float(b[i]);
        const double d = bv - av;
        an += av * av; dn += d * d; mx = std::max(mx, std::abs(d));
    }
    return {std::sqrt(dn / std::max(an, 1.0e-30)), mx};
}
std::vector<__half> reference(const std::vector<__half>& x, const std::vector<__half>& gamma) {
    std::vector<__half> y(x.size());
    for (int token = 0; token < kTokens; ++token) {
        float sum = 0.0F;
        for (int i = 0; i < kHidden; ++i) { const float v = __half2float(x[token * kHidden + i]); sum += v * v; }
        const float inv = 1.0F / std::sqrt(sum / kHidden + kEpsilon);
        for (int i = 0; i < kHidden; ++i)
            y[token * kHidden + i] = __float2half(__half2float(x[token * kHidden + i]) * inv * __half2float(gamma[i]));
    }
    return y;
}

nvinfer1::Dims dims(std::initializer_list<int32_t> values) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}

TrtPtr<nvinfer1::IHostMemory> buildPrimitive(Logger& logger, nvinfer1::IBuilder& builder) {
    const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder.createNetworkV2(flags));
    TrtPtr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
    require(network && config, "primitive network/config creation failed");
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto* x = network->addInput("X", nvinfer1::DataType::kHALF, dims({1, kTokens, kHidden}));
    auto* gamma = network->addInput("gamma", nvinfer1::DataType::kHALF, dims({kHidden}));
    require(x && gamma, "primitive input creation failed");
    auto* square = network->addElementWise(*x, *x, nvinfer1::ElementWiseOperation::kPROD);
    require(square, "primitive square failed");
    square->setName("model.layers.0.input_layernorm.square");
    const uint32_t lastAxis = 1U << 2;
    auto* mean = network->addReduce(*square->getOutput(0), nvinfer1::ReduceOperation::kAVG, lastAxis, true);
    require(mean, "primitive reduce mean failed");
    mean->setName("model.layers.0.input_layernorm.reduce_mean");
    static const __half epsilonValue = __float2half(kEpsilon);
    auto* epsilon = network->addConstant(dims({1, 1, 1}), nvinfer1::Weights{nvinfer1::DataType::kHALF, &epsilonValue, 1});
    require(epsilon, "primitive epsilon failed");
    auto* sum = network->addElementWise(*mean->getOutput(0), *epsilon->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    require(sum, "primitive epsilon add failed");
    auto* root = network->addUnary(*sum->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    require(root, "primitive sqrt failed");
    auto* normalized = network->addElementWise(*x, *root->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
    require(normalized, "primitive divide failed");
    auto* gammaReshape = network->addShuffle(*gamma);
    require(gammaReshape, "primitive gamma reshape failed");
    gammaReshape->setReshapeDimensions(dims({1, 1, kHidden}));
    auto* output = network->addElementWise(*normalized->getOutput(0), *gammaReshape->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    require(output, "primitive gamma multiply failed");
    output->setName("model.layers.0.input_layernorm");
    output->getOutput(0)->setName("original_rmsnorm_output");
    network->markOutput(*output->getOutput(0));
    output->getOutput(0)->setType(nvinfer1::DataType::kHALF);
    return TrtPtr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(*network, *config));
}

TrtPtr<nvinfer1::IHostMemory> buildPlugin(Logger& logger, nvinfer1::IBuilder& builder,
                                           nvinfer1::IPluginCreatorV3One& creator) {
    const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network(builder.createNetworkV2(flags));
    TrtPtr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
    require(network && config, "plugin network/config creation failed");
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto* x = network->addInput("X", nvinfer1::DataType::kHALF, dims({1, kTokens, kHidden}));
    auto* gamma = network->addInput("gamma", nvinfer1::DataType::kHALF, dims({kHidden}));
    require(x && gamma, "plugin input creation failed");
    const nvinfer1::PluginField field{"epsilon", &kEpsilon, nvinfer1::PluginFieldType::kFLOAT32, 1};
    const nvinfer1::PluginFieldCollection fields{1, &field};
    TrtPtr<nvinfer1::IPluginV3> plugin(creator.createPlugin("model.layers.0.input_layernorm", &fields,
                                                            nvinfer1::TensorRTPhase::kBUILD));
    require(plugin.get() != nullptr, "plugin creation failed");
    nvinfer1::ITensor* inputs[] = {x, gamma};
    auto* layer = network->addPluginV3(inputs, 2, nullptr, 0, *plugin);
    require(layer && layer->getOutput(0), "plugin layer creation failed");
    layer->setName("model.layers.0.input_layernorm.plugin");
    layer->getOutput(0)->setName("plugin_rmsnorm_output");
    network->markOutput(*layer->getOutput(0));
    layer->getOutput(0)->setType(nvinfer1::DataType::kHALF);
    return TrtPtr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(*network, *config));
}

struct RunResult { std::vector<__half> output; std::vector<float> trials; };
RunResult runEngine(nvinfer1::ICudaEngine& engine, const std::vector<__half>& x,
                    const std::vector<__half>& gamma, Logger& logger) {
    TrtPtr<nvinfer1::IExecutionContext> context(engine.createExecutionContext());
    require(context.get() != nullptr, "execution context creation failed");
    Buffer xb(sizeof(__half) * x.size()), gb(sizeof(__half) * gamma.size()), yb(sizeof(__half) * x.size());
    check(cudaMemcpy(xb.ptr, x.data(), sizeof(__half) * x.size(), cudaMemcpyHostToDevice), "copy X");
    check(cudaMemcpy(gb.ptr, gamma.data(), sizeof(__half) * gamma.size(), cudaMemcpyHostToDevice), "copy gamma");
    cudaStream_t stream{}; check(cudaStreamCreate(&stream), "stream create");
    require(context->setTensorAddress("X", xb.ptr), "set X address");
    require(context->setTensorAddress("gamma", gb.ptr), "set gamma address");
    const char* outputName = nullptr;
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        const char* name = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) outputName = name;
    }
    require(outputName && context->setTensorAddress(outputName, yb.ptr), "set output address");
    require(context->enqueueV3(stream), "initial enqueue");
    check(cudaStreamSynchronize(stream), "initial synchronize");
    for (int i = 0; i < kWarmup; ++i) require(context->enqueueV3(stream), "warmup enqueue");
    check(cudaStreamSynchronize(stream), "warmup synchronize");
    cudaEvent_t start{}, stop{}; check(cudaEventCreate(&start), "event start"); check(cudaEventCreate(&stop), "event stop");
    std::vector<float> trials;
    for (int t = 0; t < kTrials; ++t) {
        check(cudaEventRecord(start, stream), "event record start");
        for (int i = 0; i < kRepetitions; ++i) require(context->enqueueV3(stream), "timed enqueue");
        check(cudaEventRecord(stop, stream), "event record stop"); check(cudaEventSynchronize(stop), "event synchronize");
        float ms{}; check(cudaEventElapsedTime(&ms, start, stop), "event elapsed"); trials.push_back(ms / kRepetitions);
    }
    std::vector<__half> output(x.size());
    check(cudaMemcpyAsync(output.data(), yb.ptr, sizeof(__half) * output.size(), cudaMemcpyDeviceToHost, stream), "copy output");
    check(cudaStreamSynchronize(stream), "output synchronize");
    cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
    return {std::move(output), std::move(trials)};
}

double mean(const std::vector<float>& values) { double x = 0; for (float v : values) x += v; return x / values.size(); }
double stdev(const std::vector<float>& values) { const double m = mean(values); double x = 0; for (float v : values) x += (v-m)*(v-m); return std::sqrt(x / values.size()); }

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: qwen3_rmsnorm_integration <plugin.so> <x.bin> <gamma.bin> <out-prefix>");
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (library == nullptr) {
            const char* detail = dlerror();
            throw std::runtime_error(std::string("dlopen failed: ") + (detail ? detail : "unknown"));
        }
        Logger logger;
        auto* creatorInterface = getPluginRegistry()->getCreator(kPluginName, kPluginVersion, "");
        require(creatorInterface != nullptr, "RMSNormPlugin was not registered");
        auto& creator = *static_cast<nvinfer1::IPluginCreatorV3One*>(creatorInterface);
        const auto x = readHalfFile(argv[2], kElements);
        const auto gamma = readHalfFile(argv[3], kHidden);
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger)); require(builder.get() != nullptr, "builder creation failed");
        auto primitiveSerialized = buildPrimitive(logger, *builder);
        auto pluginSerialized = buildPlugin(logger, *builder, creator);
        require(primitiveSerialized && pluginSerialized, "engine serialization failed");
        std::ofstream(argv[4] + std::string("_original.engine"), std::ios::binary).write(
            static_cast<const char*>(primitiveSerialized->data()), primitiveSerialized->size());
        std::ofstream(argv[4] + std::string("_plugin.engine"), std::ios::binary).write(
            static_cast<const char*>(pluginSerialized->data()), pluginSerialized->size());
        TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger)); require(runtime.get() != nullptr, "runtime creation failed");
        TrtPtr<nvinfer1::ICudaEngine> original(runtime->deserializeCudaEngine(primitiveSerialized->data(), primitiveSerialized->size()));
        TrtPtr<nvinfer1::ICudaEngine> plugin(runtime->deserializeCudaEngine(pluginSerialized->data(), pluginSerialized->size()));
        require(original && plugin, "engine deserialization failed");
        const auto pluginRun = runEngine(*plugin, x, gamma, logger);
        const auto originalRun = runEngine(*original, x, gamma, logger);
        const auto metrics = compare(originalRun.output, pluginRun.output);
        const auto oracle = reference(x, gamma);
        const auto originalOracle = compare(oracle, originalRun.output);
        const auto pluginOracle = compare(oracle, pluginRun.output);
        std::cerr << "debug x0=" << __half2float(x[0]) << " gamma0=" << __half2float(gamma[0])
                  << " original0=" << __half2float(originalRun.output[0])
                  << " plugin0=" << __half2float(pluginRun.output[0]) << " oracle0=" << __half2float(oracle[0]) << '\n';
        std::cout << std::fixed << std::setprecision(10)
                  << "{\n  \"qwen3_node\": \"model.layers.0.input_layernorm\",\n"
                  << "  \"checkpoint_input\": true,\n  \"shape\": [1, 8, 1024],\n"
                  << "  \"epsilon\": 0.000001,\n  \"plugin_registered\": true,\n"
                  << "  \"original_engine_built\": true,\n  \"plugin_engine_built\": true,\n"
                  << "  \"serialization\": true,\n  \"deserialization\": true,\n  \"inference\": true,\n"
                  << "  \"relative_l2_error\": " << metrics.relativeL2 << ",\n"
                  << "  \"max_abs_error\": " << metrics.maxAbs << ",\n"
                  << "  \"original_vs_reference_relative_l2\": " << originalOracle.relativeL2 << ",\n"
                  << "  \"plugin_vs_reference_relative_l2\": " << pluginOracle.relativeL2 << ",\n"
                  << "  \"original_mean_latency_ms\": " << mean(originalRun.trials) << ",\n"
                  << "  \"plugin_mean_latency_ms\": " << mean(pluginRun.trials) << ",\n"
                  << "  \"original_stddev_latency_ms\": " << stdev(originalRun.trials) << ",\n"
                  << "  \"plugin_stddev_latency_ms\": " << stdev(pluginRun.trials) << ",\n"
                  << "  \"original_trial_latency_ms\": [";
        for (size_t i = 0; i < originalRun.trials.size(); ++i) std::cout << originalRun.trials[i] << (i + 1 == originalRun.trials.size() ? "" : ", ");
        std::cout << "],\n  \"plugin_trial_latency_ms\": [";
        for (size_t i = 0; i < pluginRun.trials.size(); ++i) std::cout << pluginRun.trials[i] << (i + 1 == pluginRun.trials.size() ? "" : ", ");
        std::cout << "]\n}\n";
        return metrics.relativeL2 <= 1.0e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Phase 8.3-B failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
