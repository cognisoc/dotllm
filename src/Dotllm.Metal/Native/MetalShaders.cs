namespace Dotllm.Metal.Native;

internal static class MetalShaders
{
    public const string RmsNorm = @"
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
    if (gid >= hiddenSize) return;
    float ss = 0.0;
    for (uint i = 0; i < hiddenSize; i++)
        ss += input[i] * input[i];
    float inv_norm = 1.0 / sqrt(ss / float(hiddenSize) + epsilon);
    output[gid] = weights[gid] * (input[gid] * inv_norm);
}
";

    public const string Softmax = @"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device float* scores [[buffer(0)]],
    constant float& softcap [[buffer(1)]],
    constant uint& seqLen [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= seqLen) return;
    if (softcap > 0.0)
        scores[gid] = softcap * tanh(scores[gid] / softcap);
    float max_val = -3.402823e+38;
    for (uint i = 0; i < seqLen; i++)
        max_val = max(max_val, scores[i]);
    float thread_sum = 0.0;
    for (uint i = 0; i < seqLen; i++)
    {
        float val = exp(scores[i] - max_val);
        scores[i] = val;
        thread_sum += val;
    }
    float inv_sum = 1.0 / thread_sum;
    for (uint i = 0; i < seqLen; i++)
        scores[i] *= inv_sum;
}
";

    public const string Add = @"
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
";

    public const string AddInPlace = @"
#include <metal_stdlib>
using namespace metal;

kernel void add_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    a[gid] += b[gid];
}
";

    public const string Scale = @"
#include <metal_stdlib>
using namespace metal;

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
";

    public const string ScaleInPlace = @"
#include <metal_stdlib>
using namespace metal;

kernel void scale_inplace(
    device float* data [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    data[gid] *= scalar;
}
";

    public const string Mul = @"
#include <metal_stdlib>
using namespace metal;

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
";

    public const string Silu = @"
#include <metal_stdlib>
using namespace metal;

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = input[gid];
    result[gid] = x / (1.0 + exp(-x));
}
";

    public const string SiluInPlace = @"
#include <metal_stdlib>
using namespace metal;

kernel void silu_inplace(
    device float* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = data[gid];
    data[gid] = x / (1.0 + exp(-x));
}
";

    public const string Gelu = @"
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = input[gid];
    result[gid] = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.04715 * x * x)));
}
";

    public const string MatMulF32 = @"
#include <metal_stdlib>
using namespace metal;

// MatMul: result[aRow, bCol] = sum(a[aRow, k] * b[k, bCol])
// a is [1 x aCols] (single row for inference), b is [aCols x bCols], result is [1 x bCols]
// Each thread computes one output element.
// Uses threadgroup shared memory for tiling.
kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    float dot = 0.0;
    for (uint k = 0; k < aCols; k++)
        dot += a[k] * b[k * bCols + gid];
    result[gid] = dot;
}
";

    public const string LayerNorm = @"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& hiddenSize [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hiddenSize) return;
    float mean = 0.0;
    for (uint i = 0; i < hiddenSize; i++)
        mean += input[i];
    mean /= float(hiddenSize);

    float variance = 0.0;
    for (uint i = 0; i < hiddenSize; i++)
    {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= float(hiddenSize);

    float inv_std = 1.0 / sqrt(variance + epsilon);
    output[gid] = weights[gid] * (input[gid] - mean) * inv_std + bias[gid];
}
";

    public const string DequantizeF16 = @"
#include <metal_stdlib>
using namespace metal;

// Decode F16 (IEEE 754 half) to F32
static inline float f16_to_f32(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;

    if (exponent == 0)
    {
        if (mantissa == 0)
            return sign ? -0.0 : 0.0;
        // Denormalized
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0)
            return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }

    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

// Dequantize F16 to F32, one row at a time
// src: F16 data, dst: F32 output, rowElements: number of float elements per row
kernel void dequantize_f16(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& rowElements [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rowElements) return;
    uint16_t h = *(device const uint16_t*)(src + gid * 2);
    dst[gid] = f16_to_f32(h);
}
";

    public const string DequantizeQ4_0 = @"
#include <metal_stdlib>
using namespace metal;

// Q4_0 block: float16 d; uint8_t qs[QK4_0/2]
// Each block of QK4_0=32 floats uses d (2 bytes) + qs (16 bytes) = 18 bytes
constant uint QK4_0 = 32;

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0;

kernel void dequantize_q4_0(
    device const block_q4_0* blocks [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& numRows [[buffer(2)]],
    constant uint& rowElements [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rowElements) return;
    uint block_idx = gid / QK4_0;
    uint elem_in_block = gid % QK4_0;
    float d = float(blocks[block_idx].d);
    uint q_idx = elem_in_block / 2;
    uchar qs = blocks[block_idx].qs[q_idx];
    if (elem_in_block % 2 == 0)
        dst[gid] = (float(qs & 0x0F) - 8.0f) * d;
    else
        dst[gid] = (float((qs >> 4) & 0x0F) - 8.0f) * d;
}
";

    public const string DequantizeQ8_0 = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK8_0 = 32;

typedef struct {
    half d;
    int8_t qs[QK8_0];
} block_q8_0;

kernel void dequantize_q8_0(
    device const block_q8_0* blocks [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& numRows [[buffer(2)]],
    constant uint& rowElements [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rowElements) return;
    uint block_idx = gid / QK8_0;
    uint elem_in_block = gid % QK8_0;
    float d = float(blocks[block_idx].d);
    dst[gid] = float(blocks[block_idx].qs[elem_in_block]) * d;
}
";
}