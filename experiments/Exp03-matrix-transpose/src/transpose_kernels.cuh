#pragma once

#include <cstddef>

enum class TransposeVersion { V1Copy, V2Naive, V3Tiled, V4Padded };

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

const char* versionName(TransposeVersion version);
void launchTranspose(TransposeVersion version, const float* input,
                     float* output, std::size_t width, std::size_t height,
                     void* stream = nullptr);
