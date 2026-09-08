#include "rmsnorm_plugin.h"

#include <cmath>
#include <cstring>
#include <new>

namespace phase8 {

RMSNormPluginCreator::RMSNormPluginCreator() {
    mFields.emplace_back("epsilon", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
    mFieldCollection.fields = mFields.data();
}

const char* RMSNormPluginCreator::getPluginName() const noexcept {
    return kRMSNormPluginName;
}

const char* RMSNormPluginCreator::getPluginVersion() const noexcept {
    return kRMSNormPluginVersion;
}

const char* RMSNormPluginCreator::getPluginNamespace() const noexcept {
    return "";
}

nvinfer1::PluginFieldCollection const* RMSNormPluginCreator::getFieldNames() noexcept {
    return &mFieldCollection;
}

nvinfer1::IPluginV3* RMSNormPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fields,
    nvinfer1::TensorRTPhase phase) noexcept {
    static_cast<void>(phase);
    if (name == nullptr) {
        return nullptr;
    }

    float epsilon = kDefaultEpsilon;
    if (fields != nullptr) {
        for (int32_t index = 0; index < fields->nbFields; ++index) {
            const nvinfer1::PluginField& field = fields->fields[index];
            if (field.name != nullptr && std::strcmp(field.name, "epsilon") == 0) {
                if (field.type != nvinfer1::PluginFieldType::kFLOAT32 ||
                    field.length != 1 || field.data == nullptr) {
                    return nullptr;
                }
                epsilon = *static_cast<const float*>(field.data);
            }
        }
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return nullptr;
    }
    return new (std::nothrow) RMSNormPlugin(epsilon);
}

}  // namespace phase8

using RMSNormPluginCreatorForRegistration = phase8::RMSNormPluginCreator;
REGISTER_TENSORRT_PLUGIN(RMSNormPluginCreatorForRegistration);
