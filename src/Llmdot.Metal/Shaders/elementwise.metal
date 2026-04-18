// Element-wise Metal compute shaders
// add, scale, mul, silu

#include <metal_stdlib>
using namespace metal;

kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    result[gid] = a[gid] + b[gid];
}

kernel void scale(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    result[gid] = input[gid] * scalar;
}

kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    result[gid] = a[gid] * b[gid];
}

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    result[gid] = input[gid] / (1.0 + exp(-input[gid]));
}