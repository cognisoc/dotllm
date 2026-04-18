// RmsNorm Metal compute shader
// Input: input[hiddenSize], weights[hiddenSize]
// Output: output[hiddenSize]
// Uniform: epsilon, hiddenSize

#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant uint& hiddenSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    // This is a simplified single-thread kernel.
    // Production implementation would use threadgroup reduction.
    if (gid >= hiddenSize) return;

    float ss = 0.0;
    for (uint i = 0; i < hiddenSize; i++)
        ss += input[i] * input[i];

    float inv_norm = 1.0 / sqrt(ss / float(hiddenSize) + epsilon);
    output[gid] = weights[gid] * (input[gid] * inv_norm);
}

kernel void rms_norm_threadgroup(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant uint& hiddenSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hiddenSize) return;

    float ss = 0.0;
    for (uint i = 0; i < hiddenSize; i++)
        ss += input[i] * input[i];

    float inv_norm = 1.0 / sqrt(ss / float(hiddenSize) + epsilon);
    output[gid] = weights[gid] * (input[gid] * inv_norm);
}