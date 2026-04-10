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
            case GgmlType.Q4_0:
                DequantizeQ4_0.Dequantize(src, dst, numRows, rowElements);
                break;
            case GgmlType.Q8_0:
                DequantizeQ8_0.Dequantize(src, dst, numRows, rowElements);
                break;
            default:
                DequantizeGeneric(src, dst, type, numRows, rowElements);
                break;
        }
    }

    public static void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
    {
        var aRows = result.Length / bCols;
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
                    var dot = 0f;
                    for (var k = 0; k < aCols; k++)
                        dot += aRow[k] * tmp[k];
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

    public static void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        var aRows = result.Length / bCols;
        for (var r = 0; r < aRows; r++)
        {
            var aRow = a.Slice(r * aCols, aCols);
            for (var c = 0; c < bCols; c++)
            {
                var dot = 0f;
                for (var k = 0; k < aCols; k++)
                    dot += aRow[k] * b[k * bCols + c];
                result[r * bCols + c] = dot;
            }
        }
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