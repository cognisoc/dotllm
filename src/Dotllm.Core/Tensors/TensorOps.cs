using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Dotllm.Loading;
using Dotllm.Tensors.Dequantize;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Tensors;

internal static class TensorOps
{
    public static void DequantizeToFloat(ReadOnlySpan<byte> src, Span<float> dst, GgmlType type, int numRows, int rowElements)
    {
        switch (type)
        {
            case GgmlType.F32:
                DequantizeF32.Dequantize(src, dst, numRows * rowElements);
                break;
            case GgmlType.F16:
                DequantizeF16.Dequantize(src, dst, numRows * rowElements);
                break;
            case GgmlType.BF16:
                DequantizeBF16.Dequantize(src, dst, numRows * rowElements);
                break;
            case GgmlType.Q4_0:
                DequantizeQ4_0.Dequantize(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q8_0:
                DequantizeQ8_0.Dequantize(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q2_K:
                DequantizeK.DequantizeQ2_K(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q3_K:
                DequantizeK.DequantizeQ3_K(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q4_K:
                DequantizeK.DequantizeQ4_K(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q5_K:
                DequantizeK.DequantizeQ5_K(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q6_K:
                DequantizeK.DequantizeQ6_K(src, dst, numRows, rowElements);
                break;
            default:
                DequantizeGeneric(src, dst, type, numRows, rowElements);
                break;
        }
    }

    public static void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
    {
        var aRows = result.Length / bCols;

        if (aRows == 1 && (bType == GgmlType.Q4_0 || bType == GgmlType.Q8_0))
        {
            MatMulFusedQuantized(a, b, result, bType, aCols, bCols);
            return;
        }

        float[]? rented = null;
        Span<float> tmp = aCols <= 4096 ? stackalloc float[aCols] : (rented = System.Buffers.ArrayPool<float>.Shared.Rent(aCols));

        try
        {
            for (var r = 0; r < aRows; r++)
            {
                var aRow = a.Slice(r * aCols, aCols);
                for (var c = 0; c < bCols; c++)
                {
                    DequantizeRow(b, tmp, bType, c, aCols);
                    var dot = DotProduct(aRow, tmp);
                    result[r * bCols + c] = dot;
                }
            }
        }
        finally
        {
            if (rented is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static void MatMulFusedQuantized(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
    {
        switch (bType)
        {
            case GgmlType.Q4_0:
                MatMulFusedQ4_0(a, b, result, aCols, bCols);
                break;
            case GgmlType.Q8_0:
                MatMulFusedQ8_0(a, b, result, aCols, bCols);
                break;
        }
    }

    private static void MatMulFusedQ4_0(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int blockSize = 32;
        const int halfBlock = blockSize / 2;
        const int blockBytes = 2 + halfBlock;
        var numBlocks = aCols / blockSize;
        var vecSize = Vector<float>.Count;

        Span<float> fused = aCols <= 4096 ? stackalloc float[aCols] : new float[aCols];

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockBytes;
            var fusedIdx = 0;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockBytes);
                var d = HalfHelper.HalfToFloat(b, off);

                for (var i = 0; i < halfBlock; i++)
                {
                    var qs = b[off + 2 + i];
                    var v0 = ((qs & 0x0F) - 8f) * d;
                    var v1 = (((qs >> 4) & 0x0F) - 8f) * d;

                    fused[fusedIdx] = a[fusedIdx] * v0;
                    fused[fusedIdx + 1] = a[fusedIdx + 1] * v1;
                    fusedIdx += 2;
                }
            }

            var accVec = Vector<float>.Zero;
            var k = 0;
            for (; k <= aCols - vecSize; k += vecSize)
            {
                var v = new Vector<float>(fused.Slice(k, vecSize));
                accVec += v;
            }

            var dot = Vector.Dot(accVec, Vector<float>.One);
            for (; k < aCols; k++)
                dot += fused[k];

            result[c] = dot;
        }
    }

    private static void MatMulFusedQ8_0(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int blockSize = 32;
        const int blockBytes = 2 + blockSize;
        var numBlocks = aCols / blockSize;
        var vecSize = Vector<float>.Count;

        Span<float> fused = aCols <= 4096 ? stackalloc float[aCols] : new float[aCols];

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockBytes;
            var aIdx = 0;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockBytes);
                var d = HalfHelper.HalfToFloat(b, off);

                for (var i = 0; i < blockSize; i++)
                {
                    fused[aIdx] = a[aIdx] * ((sbyte)b[off + 2 + i] * d);
                    aIdx++;
                }
            }

            var accVec = Vector<float>.Zero;
            var k = 0;
            for (; k <= aCols - vecSize; k += vecSize)
            {
                var v = new Vector<float>(fused.Slice(k, vecSize));
                accVec += v;
            }

            var dot = Vector.Dot(accVec, Vector<float>.One);
            for (; k < aCols; k++)
                dot += fused[k];

            result[c] = dot;
        }
    }

    public static void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        var aRows = result.Length / bCols;
        var vecSize = Vector<float>.Count;

        if (aRows == 1)
        {
            MatMulF32SingleRow(a, b, result, aCols, bCols);
            return;
        }

        for (var r = 0; r < aRows; r++)
        {
            var aRow = a.Slice(r * aCols, aCols);

            for (var c = 0; c < bCols; c++)
            {
                var dot = 0f;
                var accVec = Vector<float>.Zero;
                var k = 0;

                for (; k <= aCols - vecSize; k += vecSize)
                {
                    var va = new Vector<float>(aRow.Slice(k, vecSize));
                    var vb = new Vector<float>(b.Slice(k * bCols + c, vecSize));
                    accVec += va * vb;
                }

                dot = Vector.Dot(accVec, Vector<float>.One);

                for (; k < aCols; k++)
                    dot += aRow[k] * b[k * bCols + c];

                result[r * bCols + c] = dot;
            }
        }
    }

    private static void MatMulF32SingleRow(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        var tileCols = 64;
        var vecSize = Vector<float>.Count;

        for (var cStart = 0; cStart < bCols; cStart += tileCols)
        {
            var cEnd = Math.Min(cStart + tileCols, bCols);

            for (var c = cStart; c < cEnd; c++)
            {
                var accVec = Vector<float>.Zero;
                var k = 0;

                for (; k <= aCols - vecSize; k += vecSize)
                {
                    var va = new Vector<float>(a.Slice(k, vecSize));
                    var vb = new Vector<float>(b.Slice(k * bCols + c, vecSize));
                    accVec += va * vb;
                }

                var dot = Vector.Dot(accVec, Vector<float>.One);
                for (; k < aCols; k++)
                    dot += a[k] * b[k * bCols + c];

                result[c] = dot;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var n = a.Length;
        var vecSize = Vector<float>.Count;
        var accVec = Vector<float>.Zero;
        var i = 0;

        for (; i <= n - vecSize; i += vecSize)
        {
            var va = new Vector<float>(a.Slice(i, vecSize));
            var vb = new Vector<float>(b.Slice(i, vecSize));
            accVec += va * vb;
        }

        var dot = Vector.Dot(accVec, Vector<float>.One);

        for (; i < n; i++)
            dot += a[i] * b[i];

        return dot;
    }

    private static void DequantizeRow(ReadOnlySpan<byte> src, Span<float> dst, GgmlType type, int row, int rowElements)
    {
        var byteOffset = (int)((long)row * (long)TensorSize.ByteCount(type, (ulong)rowElements));
        var slice = src.Slice(byteOffset);
        switch (type)
        {
            case GgmlType.F32:
                DequantizeF32.Dequantize(slice, dst, rowElements);
                break;
            case GgmlType.F16:
                DequantizeF16.Dequantize(slice, dst, rowElements);
                break;
            case GgmlType.Q4_0:
                DequantizeQ4_0.Dequantize(slice, dst, 1, rowElements);
                break;
            case GgmlType.Q8_0:
                DequantizeQ8_0.Dequantize(slice, dst, 1, rowElements);
                break;
            default:
                DequantizeGeneric(slice, dst, type, 1, rowElements);
                break;
        }
    }

    private static void DequantizeGeneric(ReadOnlySpan<byte> src, Span<float> dst, GgmlType type, int numRows, int rowElements)
    {
        switch (type)
        {
            case GgmlType.Q4_1:
                {
                    var blockElements = 32;
                    var blockSizeBytes = 20;
                    var blocksPerRow = rowElements / blockElements;
                    for (var r = 0; r < numRows; r++)
                    for (var b = 0; b < blocksPerRow; b++)
                    {
                        var off = r * blocksPerRow * blockSizeBytes + b * blockSizeBytes;
                        var d = HalfHelper.HalfToFloat(src, off);
                        var m = HalfHelper.HalfToFloat(src, off + 2);
                        for (var i = 0; i < blockElements; i++)
                        {
                            var nibble = (src[off + 4 + i / 2] >> ((i & 1) * 4)) & 0x0F;
                            dst[r * rowElements + b * blockElements + i] = d * nibble - m * 8;
                        }
                    }
                    break;
                }
            case GgmlType.Q5_0:
                {
                    var blockElements = 32;
                    var blockSizeBytes = 22;
                    var blocksPerRow = rowElements / blockElements;
                    for (var r = 0; r < numRows; r++)
                    for (var b = 0; b < blocksPerRow; b++)
                    {
                        var off = r * blocksPerRow * blockSizeBytes + b * blockSizeBytes;
                        var d = HalfHelper.HalfToFloat(src, off);
                        var qh = src[off + 2];
                        for (var i = 0; i < blockElements; i++)
                        {
                            var byteIdx = 3 + i / 2;
                            var nibble = (src[off + byteIdx] >> ((i & 1) * 4)) & 0x0F;
                            var high = (qh >> i) & 1;
                            var val = nibble + high * 16;
                            dst[r * rowElements + b * blockElements + i] = (val - 16) * d;
                        }
                    }
                    break;
                }
            case GgmlType.Q5_1:
                {
                    var blockElements = 32;
                    var blockSizeBytes = 24;
                    var blocksPerRow = rowElements / blockElements;
                    for (var r = 0; r < numRows; r++)
                    for (var b = 0; b < blocksPerRow; b++)
                    {
                        var off = r * blocksPerRow * blockSizeBytes + b * blockSizeBytes;
                        var d = HalfHelper.HalfToFloat(src, off);
                        var m = HalfHelper.HalfToFloat(src, off + 2);
                        var qh = src[off + 4];
                        for (var i = 0; i < blockElements; i++)
                        {
                            var byteIdx = 5 + i / 2;
                            var nibble = (src[off + byteIdx] >> ((i & 1) * 4)) & 0x0F;
                            var high = (qh >> i) & 1;
                            var val = nibble + high * 16;
                            dst[r * rowElements + b * blockElements + i] = d * val - m * 16;
                        }
                    }
                    break;
                }
            default:
                throw new NotSupportedException($"Dequantization for type {type} not yet implemented.");
        }
    }

    public static void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim)
    {
        var actualRotaryDim = rotaryDim > 0 ? rotaryDim : headDim;
        var halfDim = actualRotaryDim / 2;

        for (var i = 0; i < halfDim; i++)
        {
            var freq = 1f / MathF.Pow(freqBase, 2f * i / actualRotaryDim);
            var angle = position * freq;
            var cosVal = MathF.Cos(angle);
            var sinVal = MathF.Sin(angle);

            var qIdx = i;
            var qHalfIdx = i + halfDim;
            var q0 = query[qIdx];
            var q1 = query[qHalfIdx];
            query[qIdx] = q0 * cosVal - q1 * sinVal;
            query[qHalfIdx] = q0 * sinVal + q1 * cosVal;

            var k0 = key[qIdx];
            var k1 = key[qHalfIdx];
            key[qIdx] = k0 * cosVal - k1 * sinVal;
            key[qHalfIdx] = k0 * sinVal + k1 * cosVal;
        }
    }

    public static void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, ReadOnlySpan<float> freqTable)
    {
        var halfDim = freqTable.Length;

        for (var i = 0; i < halfDim; i++)
        {
            var angle = position * freqTable[i];
            var cosVal = MathF.Cos(angle);
            var sinVal = MathF.Sin(angle);

            var qIdx = i;
            var qHalfIdx = i + halfDim;
            var q0 = query[qIdx];
            var q1 = query[qHalfIdx];
            query[qIdx] = q0 * cosVal - q1 * sinVal;
            query[qHalfIdx] = q0 * sinVal + q1 * cosVal;

            var k0 = key[qIdx];
            var k1 = key[qHalfIdx];
            key[qIdx] = k0 * cosVal - k1 * sinVal;
            key[qHalfIdx] = k0 * sinVal + k1 * cosVal;
        }
    }

    public static void Conv1D(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int kernelSize, int inputDim)
    {
        for (var i = 0; i < inputDim; i++)
        {
            var sum = 0f;
            for (var k = 0; k < kernelSize; k++)
                sum += input[k * inputDim + i] * weights[k * inputDim + i];
            output[i] = sum;
        }
    }
}