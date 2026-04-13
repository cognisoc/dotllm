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

    public const string MatMulQ4_0 = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK4_0 = 32;
constant uint QK4_0_HALF = 16;

typedef struct {
    half d;
    uchar qs[QK4_0_HALF];
} block_q4_0;

kernel void matmul_q4_0(
    device const float* a [[buffer(0)]],
    device const block_q4_0* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint numBlocks = aCols / QK4_0;
    ulong rowOffset = (ulong)gid * numBlocks * sizeof(block_q4_0);
    device const block_q4_0* rowBlocks = (device const block_q4_0*)((device const uchar*)b + rowOffset);

    float dot = 0.0;
    for (uint blk = 0; blk < numBlocks; blk++)
    {
        float d = float(rowBlocks[blk].d);
        uint baseIdx = blk * QK4_0;
        for (uint i = 0; i < QK4_0_HALF; i++)
        {
            uchar qs = rowBlocks[blk].qs[i];
            float v0 = (float(qs & 0x0F) - 8.0f) * d;
            float v1 = (float((qs >> 4) & 0x0F) - 8.0f) * d;
            dot += a[baseIdx + i * 2] * v0;
            dot += a[baseIdx + i * 2 + 1] * v1;
        }
    }
    result[gid] = dot;
}
";

    public const string MatMulQ8_0 = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK8_0 = 32;

typedef struct {
    half d;
    int8_t qs[QK8_0];
} block_q8_0;

kernel void matmul_q8_0(
    device const float* a [[buffer(0)]],
    device const block_q8_0* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint numBlocks = aCols / QK8_0;
    ulong rowOffset = (ulong)gid * numBlocks * sizeof(block_q8_0);
    device const block_q8_0* rowBlocks = (device const block_q8_0*)((device const uchar*)b + rowOffset);

    float dot = 0.0;
    for (uint blk = 0; blk < numBlocks; blk++)
    {
        float d = float(rowBlocks[blk].d);
        uint baseIdx = blk * QK8_0;
        for (uint i = 0; i < QK8_0; i++)
            dot += a[baseIdx + i] * float(rowBlocks[blk].qs[i]) * d;
    }
    result[gid] = dot;
}
";

    public const string MatMulQ4_K = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK4_K = 256;

static inline int extract6bit(device const uchar* src, uint baseOffset, uint bitOffset)
{
    uint byteIdx = bitOffset / 8;
    uint bitShift = bitOffset % 8;
    if (bitShift <= 2)
        return (src[baseOffset + byteIdx] >> bitShift) & 0x3F;
    return ((src[baseOffset + byteIdx] >> bitShift) |
            (src[baseOffset + byteIdx + 1] << (8 - bitShift))) & 0x3F;
}

static inline float f16_to_f32(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0) return sign ? -0.0 : 0.0;
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

constant uint Q4_K_BLOCK_BYTES = 144;

kernel void matmul_q4_k(
    device const float* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint blocksPerRow = aCols / QK4_K;
    ulong rowBase = (ulong)gid * blocksPerRow * Q4_K_BLOCK_BYTES;

    float dot = 0.0;
    for (uint blk = 0; blk < blocksPerRow; blk++)
    {
        ulong off = rowBase + blk * Q4_K_BLOCK_BYTES;
        uint16_t dRaw = *(device const uint16_t*)(b + off);
        uint16_t dminRaw = *(device const uint16_t*)(b + off + 2);
        float d = f16_to_f32(dRaw);
        float dmin = f16_to_f32(dminRaw);

        for (uint i = 0; i < QK4_K; i++)
        {
            uint g = i / 32;
            int sc = extract6bit(b, off + 4, g * 12);
            int mi = extract6bit(b, off + 4, g * 12 + 6);
            int q = (b[off + 16 + i / 2] >> ((i % 2) * 4)) & 0xF;
            dot += a[blk * QK4_K + i] * (d * float(sc) * float(q - 8) - dmin * float(mi));
        }
    }
    result[gid] = dot;
}
";

    public const string MatMulQ6_K = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK6_K = 256;

static inline float f16_to_f32_q6(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0) return sign ? -0.0 : 0.0;
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

constant uint Q6_K_BLOCK_BYTES = 210;

kernel void matmul_q6_k(
    device const float* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint blocksPerRow = aCols / QK6_K;
    ulong rowBase = (ulong)gid * blocksPerRow * Q6_K_BLOCK_BYTES;

    float dot = 0.0;
    for (uint blk = 0; blk < blocksPerRow; blk++)
    {
        ulong off = rowBase + blk * Q6_K_BLOCK_BYTES;
        uint16_t dRaw = *(device const uint16_t*)(b + off + 208);
        float d = f16_to_f32_q6(dRaw);

        for (uint i = 0; i < QK6_K; i++)
        {
            uint g = i / 16;
            uint j = i % 16;
            int qlVal;
            if (g < 8)
                qlVal = b[off + 16 * g + j];
            else
                qlVal = b[off + 16 * (g - 8) + j + 128];
            uint qhIdx = 4 * g + j / 4;
            uint qhShift = 2 * (j % 4);
            int qhBits = (b[off + 128 + qhIdx] >> qhShift) & 0x3;
            int q6 = (qlVal | (qhBits << 4)) - 32;
            int sc = (int)(int8_t)b[off + 192 + g];
            dot += a[blk * QK6_K + i] * d * float(sc) * float(q6);
        }
    }
    result[gid] = dot;
}
";

    public const string MatMulQ2_K = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK2_K = 256;

static inline float f16_q2(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0) return sign ? -0.0 : 0.0;
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

constant uint Q2_K_BLOCK_BYTES = 84;

kernel void matmul_q2_k(
    device const float* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint blocksPerRow = aCols / QK2_K;
    ulong rowBase = (ulong)gid * blocksPerRow * Q2_K_BLOCK_BYTES;

    float dot = 0.0;
    for (uint blk = 0; blk < blocksPerRow; blk++)
    {
        ulong off = rowBase + blk * Q2_K_BLOCK_BYTES;
        float d = f16_q2(*(device const uint16_t*)(b + off));
        float dmin = f16_q2(*(device const uint16_t*)(b + off + 2));

        for (uint i = 0; i < QK2_K; i++)
        {
            uint g = i / 64;
            uint local = i % 64;
            uint sub = local / 16;
            uint scIdx = g * 4 + sub;
            int sc = (b[off + 4 + scIdx * 2] & 0x0F);
            int mi = (b[off + 4 + scIdx * 2 + 1] & 0x0F);
            int q = (b[off + 20 + i / 4] >> ((i % 4) * 2)) & 0x3;
            dot += a[blk * QK2_K + i] * (d * float(sc) * (float(q) - 0.5f) - dmin * float(mi));
        }
    }
    result[gid] = dot;
}
";

    public const string MatMulQ3_K = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK3_K = 256;

static inline int extract6bit_q3(device const uchar* src, uint baseOffset, uint bitOffset)
{
    uint byteIdx = bitOffset / 8;
    uint bitShift = bitOffset % 8;
    if (bitShift <= 2)
        return (src[baseOffset + byteIdx] >> bitShift) & 0x3F;
    return ((src[baseOffset + byteIdx] >> bitShift) |
            (src[baseOffset + byteIdx + 1] << (8 - bitShift))) & 0x3F;
}

static inline float f16_q3(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0) return sign ? -0.0 : 0.0;
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

constant uint Q3_K_BLOCK_BYTES = 110;

kernel void matmul_q3_k(
    device const float* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint blocksPerRow = aCols / QK3_K;
    ulong rowBase = (ulong)gid * blocksPerRow * Q3_K_BLOCK_BYTES;

    float dot = 0.0;
    for (uint blk = 0; blk < blocksPerRow; blk++)
    {
        ulong off = rowBase + blk * Q3_K_BLOCK_BYTES;
        float d = f16_q3(*(device const uint16_t*)(b + off));

        for (uint i = 0; i < QK3_K; i++)
        {
            uint g = i / 16;
            int sc = extract6bit_q3(b, off + 4, g * 6);
            int q = (b[off + 48 + i / 4] >> ((i % 4) * 2)) & 0x3;
            int m = (b[off + 16 + i / 8] >> (i % 8)) & 0x1;
            int value = q - (m << 2);
            dot += a[blk * QK3_K + i] * d * float(sc) * float(value);
        }
    }
    result[gid] = dot;
}
";

    public const string MatMulQ5_K = @"
#include <metal_stdlib>
using namespace metal;

constant uint QK5_K = 256;

static inline int extract6bit_q5(device const uchar* src, uint baseOffset, uint bitOffset)
{
    uint byteIdx = bitOffset / 8;
    uint bitShift = bitOffset % 8;
    if (bitShift <= 2)
        return (src[baseOffset + byteIdx] >> bitShift) & 0x3F;
    return ((src[baseOffset + byteIdx] >> bitShift) |
            (src[baseOffset + byteIdx + 1] << (8 - bitShift))) & 0x3F;
}

static inline float f16_q5(uint16_t h)
{
    uint sign = (h >> 15) & 1;
    uint exponent = (h >> 10) & 0x1F;
    uint mantissa = h & 0x3FF;
    if (exponent == 0)
    {
        if (mantissa == 0) return sign ? -0.0 : 0.0;
        float f = mantissa / 1024.0f;
        return sign ? -f : f;
    }
    if (exponent == 31)
    {
        if (mantissa == 0) return sign ? -INFINITY : INFINITY;
        return sign ? -NAN : NAN;
    }
    float f = (1.0f + mantissa / 1024.0f) * pow(2.0f, (float)(exponent - 15));
    return sign ? -f : f;
}

constant uint Q5_K_BLOCK_BYTES = 176;

kernel void matmul_q5_k(
    device const float* a [[buffer(0)]],
    device const uchar* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& aCols [[buffer(3)]],
    constant uint& bCols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= bCols) return;
    uint blocksPerRow = aCols / QK5_K;
    ulong rowBase = (ulong)gid * blocksPerRow * Q5_K_BLOCK_BYTES;

    float dot = 0.0;
    for (uint blk = 0; blk < blocksPerRow; blk++)
    {
        ulong off = rowBase + blk * Q5_K_BLOCK_BYTES;
        float d = f16_q5(*(device const uint16_t*)(b + off));
        float dmin = f16_q5(*(device const uint16_t*)(b + off + 2));

        for (uint i = 0; i < QK5_K; i++)
        {
            uint g = i / 32;
            int sc = extract6bit_q5(b, off + 4, g * 12);
            int mi = extract6bit_q5(b, off + 4, g * 12 + 6);
            int qLow = (b[off + 48 + i / 2] >> ((i % 2) * 4)) & 0xF;
            int qHigh = (b[off + 16 + i / 8] >> (i % 8)) & 0x1;
            int q5 = qLow + (qHigh << 4);
            dot += a[blk * QK5_K + i] * (d * float(sc) * float(q5 - 16) - dmin * float(mi));
        }
    }
    result[gid] = dot;
}
";
}