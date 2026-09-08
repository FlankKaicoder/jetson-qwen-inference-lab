#include <NvInfer.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
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
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

template <typename T>
struct TrtDestroy {
    void operator()(T* object) const noexcept {
        if (object != nullptr) {
            delete object;
        }
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroy<T>>;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    void* pointer{nullptr};

    explicit DeviceBuffer(size_t bytes) {
        cudaCheck(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct ErrorMetrics {
    double relativeL2{0.0};
    double maxAbs{0.0};
};

ErrorMetrics compareWithReference(const std::vector<__half>& input,
                                  const std::vector<__half>& gamma,
                                  const std::vector<__half>& output) {
    double referenceNorm = 0.0;
    double differenceNorm = 0.0;
    double maxAbs = 0.0;
    for (int token = 0; token < kTokens; ++token) {
        float sumSquares = 0.0F;
        for (int index = 0; index < kHidden; ++index) {
            const float value = __half2float(input[token * kHidden + index]);
            sumSquares += value * value;
        }
        const float invRms = 1.0F / std::sqrt(sumSquares / kHidden + kEpsilon);
        for (int index = 0; index < kHidden; ++index) {
            const float reference = __half2float(input[token * kHidden + index]) *
                                    invRms * __half2float(gamma[index]);
            const float actual = __half2float(output[token * kHidden + index]);
            const double difference = static_cast<double>(actual) - reference;
            referenceNorm += static_cast<double>(reference) * reference;
            differenceNorm += difference * difference;
            maxAbs = std::max(maxAbs, std::abs(difference));
        }
    }
    return {std::sqrt(differenceNorm / referenceNorm), maxAbs};
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: rmsnorm_plugin_demo <plugin-library>");
        void* pluginLibrary = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (pluginLibrary == nullptr) {
            throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        }

        Logger logger;
        auto* creatorInterface = getPluginRegistry()->getCreator(
            kPluginName, kPluginVersion, "");
        require(creatorInterface != nullptr, "IPluginCreatorV3One was not registered");
        auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(creatorInterface);

        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        require(builder != nullptr, "createInferBuilder failed");
        const auto flags = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        require(network != nullptr && config != nullptr, "network or config creation failed");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        nvinfer1::Dims xDims{};
        xDims.nbDims = 3;
        xDims.d[0] = 1;
        xDims.d[1] = kTokens;
        xDims.d[2] = kHidden;
        nvinfer1::Dims gammaDims{};
        gammaDims.nbDims = 1;
        gammaDims.d[0] = kHidden;
        auto* x = network->addInput("X", nvinfer1::DataType::kHALF, xDims);
        auto* gamma = network->addInput("gamma", nvinfer1::DataType::kHALF, gammaDims);
        require(x != nullptr && gamma != nullptr, "network input creation failed");

        const nvinfer1::PluginField fields[] = {
            {"epsilon", &kEpsilon, nvinfer1::PluginFieldType::kFLOAT32, 1},
        };
        const nvinfer1::PluginFieldCollection fieldCollection{1, fields};
        TrtPtr<nvinfer1::IPluginV3> plugin(
            creator->createPlugin(kPluginName, &fieldCollection,
                                  nvinfer1::TensorRTPhase::kBUILD));
        require(plugin != nullptr, "plugin creation failed");

        nvinfer1::ITensor* inputs[] = {x, gamma};
        auto* layer = network->addPluginV3(inputs, 2, nullptr, 0, *plugin);
        require(layer != nullptr && layer->getOutput(0) != nullptr,
                "addPluginV3 failed");
        layer->getOutput(0)->setName("Y");
        network->markOutput(*layer->getOutput(0));
        layer->getOutput(0)->setType(nvinfer1::DataType::kHALF);

        TrtPtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        require(serialized != nullptr, "buildSerializedNetwork failed");
        TrtPtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        TrtPtr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(
            serialized->data(), serialized->size()));
        require(engine != nullptr, "deserializeCudaEngine failed");
        TrtPtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        require(context != nullptr, "createExecutionContext failed");

        std::mt19937 generator(20260908U);
        std::normal_distribution<float> distribution(0.0F, 1.0F);
        std::vector<__half> hostInput(kElements);
        std::vector<__half> hostGamma(kHidden);
        std::vector<__half> hostOutput(kElements);
        for (auto& value : hostInput) {
            value = __float2half(distribution(generator));
        }
        for (auto& value : hostGamma) {
            value = __float2half(distribution(generator));
        }

        DeviceBuffer inputBuffer(sizeof(__half) * hostInput.size());
        DeviceBuffer gammaBuffer(sizeof(__half) * hostGamma.size());
        DeviceBuffer outputBuffer(sizeof(__half) * hostOutput.size());
        cudaCheck(cudaMemcpy(inputBuffer.pointer, hostInput.data(),
                             sizeof(__half) * hostInput.size(),
                             cudaMemcpyHostToDevice),
                  "copy X to device");
        cudaCheck(cudaMemcpy(gammaBuffer.pointer, hostGamma.data(),
                             sizeof(__half) * hostGamma.size(),
                             cudaMemcpyHostToDevice),
                  "copy gamma to device");
        cudaStream_t stream{};
        cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");
        require(context->setTensorAddress("X", inputBuffer.pointer),
                "set X tensor address failed");
        require(context->setTensorAddress("gamma", gammaBuffer.pointer),
                "set gamma tensor address failed");
        require(context->setTensorAddress("Y", outputBuffer.pointer),
                "set Y tensor address failed");

        require(context->enqueueV3(stream), "initial enqueueV3 failed");
        cudaCheck(cudaMemcpyAsync(hostOutput.data(), outputBuffer.pointer,
                                  sizeof(__half) * hostOutput.size(),
                                  cudaMemcpyDeviceToHost, stream),
                  "copy Y to host");
        cudaCheck(cudaStreamSynchronize(stream), "initial inference synchronization");
        const ErrorMetrics error = compareWithReference(hostInput, hostGamma, hostOutput);

        for (int index = 0; index < kWarmup; ++index) {
            require(context->enqueueV3(stream), "warmup enqueueV3 failed");
        }
        cudaCheck(cudaStreamSynchronize(stream), "warmup synchronization");

        cudaEvent_t start{};
        cudaEvent_t stop{};
        cudaCheck(cudaEventCreate(&start), "cudaEventCreate start");
        cudaCheck(cudaEventCreate(&stop), "cudaEventCreate stop");
        std::vector<float> trialMilliseconds;
        for (int trial = 0; trial < kTrials; ++trial) {
            cudaCheck(cudaEventRecord(start, stream), "record start event");
            for (int repetition = 0; repetition < kRepetitions; ++repetition) {
                require(context->enqueueV3(stream), "timed enqueueV3 failed");
            }
            cudaCheck(cudaEventRecord(stop, stream), "record stop event");
            cudaCheck(cudaEventSynchronize(stop), "synchronize stop event");
            float elapsedMilliseconds = 0.0F;
            cudaCheck(cudaEventElapsedTime(&elapsedMilliseconds, start, stop),
                      "read elapsed event time");
            trialMilliseconds.push_back(elapsedMilliseconds / kRepetitions);
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);

        double mean = 0.0;
        for (float trial : trialMilliseconds) {
            mean += trial;
        }
        mean /= trialMilliseconds.size();
        double variance = 0.0;
        for (float trial : trialMilliseconds) {
            const double delta = trial - mean;
            variance += delta * delta;
        }
        variance /= trialMilliseconds.size();

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "{\n"
                  << "  \"tensorrt_version\": \"" << NV_TENSORRT_MAJOR << "."
                  << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\",\n"
                  << "  \"plugin_registered\": true,\n"
                  << "  \"engine_built\": true,\n"
                  << "  \"inference_ran\": true,\n"
                  << "  \"shape\": [1, 8, 1024],\n"
                  << "  \"dtype\": \"float16\",\n"
                  << "  \"epsilon\": " << kEpsilon << ",\n"
                  << "  \"relative_l2_error\": " << error.relativeL2 << ",\n"
                  << "  \"max_abs_error\": " << error.maxAbs << ",\n"
                  << "  \"warmup\": " << kWarmup << ",\n"
                  << "  \"repetitions_per_trial\": " << kRepetitions << ",\n"
                  << "  \"trials\": " << kTrials << ",\n"
                  << "  \"trial_latency_ms\": [";
        for (size_t index = 0; index < trialMilliseconds.size(); ++index) {
            std::cout << trialMilliseconds[index]
                      << (index + 1 == trialMilliseconds.size() ? "" : ", ");
        }
        std::cout << "],\n"
                  << "  \"mean_latency_ms\": " << mean << ",\n"
                  << "  \"stddev_latency_ms\": " << std::sqrt(variance) << "\n"
                  << "}\n";
        return error.relativeL2 <= 1.0e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Phase 8.3-A failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
