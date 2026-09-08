#pragma once

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include <vector>

namespace phase8 {

inline constexpr char kRMSNormPluginName[] = "RMSNormPlugin";
inline constexpr char kRMSNormPluginVersion[] = "1";
inline constexpr int kHiddenSize = 1024;
inline constexpr int kTokenCount = 8;
inline constexpr float kDefaultEpsilon = 1.0e-6F;

cudaError_t launchRMSNormPluginKernel(const void* input, void* output,
                                      const void* gamma, int tokens,
                                      float epsilon, cudaStream_t stream);

class RMSNormPlugin final : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime {
public:
    explicit RMSNormPlugin(float epsilon = kDefaultEpsilon);

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                            int32_t nbInputs,
                            const nvinfer1::DynamicPluginTensorDesc* outputs,
                            int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
                               int32_t nbOutputs,
                               const nvinfer1::DataType* inputTypes,
                               int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs,
                            int32_t nbInputs,
                            const nvinfer1::DimsExprs* shapeInputs,
                            int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs,
                            int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t nbOutputs) noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                    const nvinfer1::PluginTensorDesc* outputDesc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    bool hasSupportedDimensions(const nvinfer1::PluginTensorDesc* inputs,
                                int32_t nbInputs,
                                const nvinfer1::PluginTensorDesc* outputs,
                                int32_t nbOutputs) const noexcept;
    void refreshSerializationFields() noexcept;

    float mEpsilon;
    nvinfer1::PluginField mSerializationField{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class RMSNormPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    RMSNormPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mFieldCollection{};
};

}  // namespace phase8
