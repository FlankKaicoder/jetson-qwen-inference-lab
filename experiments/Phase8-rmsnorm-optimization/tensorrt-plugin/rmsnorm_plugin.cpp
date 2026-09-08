#include "rmsnorm_plugin.h"

#include <cmath>
#include <cstring>
#include <new>

namespace phase8 {
namespace {

bool isHalfLinear(const nvinfer1::PluginTensorDesc& desc) noexcept {
    return desc.type == nvinfer1::DataType::kHALF &&
           desc.format == nvinfer1::PluginFormat::kLINEAR;
}

bool isInputShape(const nvinfer1::Dims& dims) noexcept {
    return dims.nbDims == 3 && dims.d[0] != 0 && dims.d[1] != 0 &&
           dims.d[2] == kHiddenSize;
}

bool isGammaShape(const nvinfer1::Dims& dims) noexcept {
    return dims.nbDims == 1 && dims.d[0] == kHiddenSize;
}

}  // namespace

RMSNormPlugin::RMSNormPlugin(float epsilon) : mEpsilon(epsilon) {
    refreshSerializationFields();
}

nvinfer1::IPluginCapability* RMSNormPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* RMSNormPlugin::clone() noexcept {
    return new (std::nothrow) RMSNormPlugin(mEpsilon);
}

const char* RMSNormPlugin::getPluginName() const noexcept {
    return kRMSNormPluginName;
}

const char* RMSNormPlugin::getPluginVersion() const noexcept {
    return kRMSNormPluginVersion;
}

const char* RMSNormPlugin::getPluginNamespace() const noexcept {
    return "";
}

int32_t RMSNormPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t RMSNormPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int32_t nbOutputs) noexcept {
    if (inputs == nullptr || outputs == nullptr || nbInputs != 2 || nbOutputs != 1 ||
        !isHalfLinear(inputs[0].desc) || !isHalfLinear(inputs[1].desc) ||
        !isHalfLinear(outputs[0].desc) || !isInputShape(inputs[0].desc.dims) ||
        !isGammaShape(inputs[1].desc.dims) || !isInputShape(outputs[0].desc.dims)) {
        return 1;
    }
    return 0;
}

bool RMSNormPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (inOut == nullptr || nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos > 2) {
        return false;
    }
    if (!isHalfLinear(inOut[pos].desc)) {
        return false;
    }
    return pos != 2 || inOut[2].desc.type == inOut[0].desc.type;
}

int32_t RMSNormPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept {
    if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 2 ||
        inputTypes[0] != nvinfer1::DataType::kHALF ||
        inputTypes[1] != nvinfer1::DataType::kHALF) {
        return 1;
    }
    outputTypes[0] = nvinfer1::DataType::kHALF;
    return 0;
}

int32_t RMSNormPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
    const nvinfer1::DimsExprs* shapeInputs, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    static_cast<void>(shapeInputs);
    static_cast<void>(nbShapeInputs);
    static_cast<void>(exprBuilder);
    if (inputs == nullptr || outputs == nullptr || nbInputs != 2 || nbOutputs != 1 ||
        inputs[0].nbDims != 3 || inputs[1].nbDims != 1) {
        return 1;
    }
    outputs[0] = inputs[0];
    return 0;
}

int32_t RMSNormPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) noexcept {
    return hasSupportedDimensions(inputs, nbInputs, outputs, nbOutputs) ? 0 : 1;
}

int32_t RMSNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                               const nvinfer1::PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs,
                               void* workspace, cudaStream_t stream) noexcept {
    static_cast<void>(workspace);
    if (!hasSupportedDimensions(inputDesc, 2, outputDesc, 1) || inputs == nullptr ||
        outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr ||
        outputs[0] == nullptr) {
        return 1;
    }
    const int batch = inputDesc[0].dims.d[0];
    const int sequence = inputDesc[0].dims.d[1];
    if (batch <= 0 || sequence <= 0) {
        return 1;
    }
    return launchRMSNormPluginKernel(inputs[0], outputs[0], inputs[1],
                                     batch * sequence, mEpsilon, stream) == cudaSuccess
               ? 0
               : 1;
}

nvinfer1::IPluginV3* RMSNormPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* context) noexcept {
    static_cast<void>(context);
    return clone();
}

nvinfer1::PluginFieldCollection const* RMSNormPlugin::getFieldsToSerialize() noexcept {
    return &mSerializationCollection;
}

bool RMSNormPlugin::hasSupportedDimensions(
    const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept {
    return inputs != nullptr && outputs != nullptr && nbInputs == 2 && nbOutputs == 1 &&
           isHalfLinear(inputs[0]) && isHalfLinear(inputs[1]) && isHalfLinear(outputs[0]) &&
           isInputShape(inputs[0].dims) && isGammaShape(inputs[1].dims) &&
           isInputShape(outputs[0].dims);
}

void RMSNormPlugin::refreshSerializationFields() noexcept {
    mSerializationField = nvinfer1::PluginField(
        "epsilon", &mEpsilon, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mSerializationCollection.nbFields = 1;
    mSerializationCollection.fields = &mSerializationField;
}

}  // namespace phase8
