using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Buffers.Binary;
using Llmdot.Loading;
using Llmdot.Tensors.Dequantize;
using Llmdot.Tensors.Numeric;

#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace Llmdot.Tensors;

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

    #region MatMul Dispatch

    public static void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
    {
        var aRows = result.Length / bCols;

        if (aRows == 1)
        {
            switch (bType)
            {
                case GgmlType.Q4_0:
                    MatMulFusedQ4_0(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q8_0:
                    MatMulFusedQ8_0(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q6_K:
                    MatMulFusedQ6_K(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q4_K:
                    MatMulFusedQ4_K(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q2_K:
                    MatMulFusedQ2_K(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q3_K:
                    MatMulFusedQ3_K(a, b, result, aCols, bCols);
                    return;
                case GgmlType.Q5_K:
                    MatMulFusedQ5_K(a, b, result, aCols, bCols);
                    return;
            }
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
                    result[r * bCols + c] = DotProduct(aRow, tmp);
                }
            }
        }
        finally
        {
            if (rented is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }
    }

    #endregion

    #region Q4_0 Fused MatMul (Optimized)

    private static void MatMulFusedQ4_0(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int blockSize = 32;
        const int halfBlock = blockSize / 2;
        const int blockBytes = 2 + halfBlock;
        var numBlocks = aCols / blockSize;
        var vecSize = Vector<float>.Count;

        float[]? rentedBuf = null;
        Span<float> fused = aCols <= 4096 ? stackalloc float[aCols] : (rentedBuf = System.Buffers.ArrayPool<float>.Shared.Rent(aCols));
        if (rentedBuf is not null) fused = fused[..aCols];

        try
        {
            for (var c = 0; c < bCols; c++)
            {
                var rowOffset = (long)c * numBlocks * blockBytes;
                var fusedIdx = 0;

                for (var blk = 0; blk < numBlocks; blk++)
                {
                    var off = (int)(rowOffset + blk * blockBytes);
                    var d = HalfHelper.HalfToFloat(b, off);
                    var baseIdx = blk * blockSize;

                    for (var i = 0; i < halfBlock; i++)
                    {
                        var qs = b[off + 2 + i];
                        fused[baseIdx + i] = a[baseIdx + i] * (((qs & 0x0F) - 8f) * d);
                        fused[baseIdx + i + halfBlock] = a[baseIdx + i + halfBlock] * ((((qs >> 4) & 0x0F) - 8f) * d);
                    }

                    fusedIdx += blockSize;
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
        finally
        {
            if (rentedBuf is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rentedBuf);
        }
    }

    #endregion

    #region Q8_0 Fused MatMul (Optimized)

    private static void MatMulFusedQ8_0(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int blockSize = 32;
        const int blockBytes = 2 + blockSize;
        var numBlocks = aCols / blockSize;
        var vecSize = Vector<float>.Count;

        float[]? rentedBuf = null;
        Span<float> fused = aCols <= 4096 ? stackalloc float[aCols] : (rentedBuf = System.Buffers.ArrayPool<float>.Shared.Rent(aCols));
        if (rentedBuf is not null) fused = fused[..aCols];

        try
        {
            for (var c = 0; c < bCols; c++)
            {
                var rowOffset = (long)c * numBlocks * blockBytes;
                var accVec = Vector<float>.Zero;
                var fusedIdx = 0;

                for (var blk = 0; blk < numBlocks; blk++)
                {
                    var off = (int)(rowOffset + blk * blockBytes);
                    var d = HalfHelper.HalfToFloat(b, off);

                    for (var i = 0; i < blockSize; i++)
                    {
                        fused[fusedIdx] = a[fusedIdx] * ((sbyte)b[off + 2 + i] * d);
                        fusedIdx++;
                    }
                }

                var k = 0;
                for (; k <= aCols - vecSize; k += vecSize)
                    accVec += new Vector<float>(fused.Slice(k, vecSize));

                result[c] = Vector.Dot(accVec, Vector<float>.One);
                for (; k < aCols; k++)
                    result[c] += fused[k];
            }
        }
        finally
        {
            if (rentedBuf is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rentedBuf);
        }
    }

    #endregion

    #region Q6_K Fused MatMul (New)

    private static void MatMulFusedQ6_K(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int qk = 256;
        const int blockSizeBytes = 210;
        var numBlocks = aCols / qk;

        float[]? rentedBuf = null;
        Span<float> deqBuf = aCols <= 4096 ? stackalloc float[aCols] : (rentedBuf = System.Buffers.ArrayPool<float>.Shared.Rent(aCols));
        if (rentedBuf is not null) deqBuf = deqBuf[..aCols];

        try
        {
            for (var c = 0; c < bCols; c++)
            {
                var rowOffset = (long)c * numBlocks * blockSizeBytes;

                for (var blk = 0; blk < numBlocks; blk++)
                {
                    var off = (int)(rowOffset + blk * blockSizeBytes);
                    var d = HalfHelper.HalfToFloat(b, off + 208);

                    for (var n = 0; n < qk; n += 128)
                    {
                        var dstBase = blk * qk + n;
                        var qlBase = off + n / 2;
                        var qhBase = off + 128 + n / 4;
                        var scBase = off + 192 + (n / 16);

                        for (var l = 0; l < 32; l++)
                        {
                            var isIdx = l / 16;
                            var q1 = ((b[qlBase + l] & 0xF) | (((b[qhBase + l] >> 0) & 3) << 4)) - 32;
                            var q2 = ((b[qlBase + l + 32] & 0xF) | (((b[qhBase + l] >> 2) & 3) << 4)) - 32;
                            var q3 = ((b[qlBase + l] >> 4) | (((b[qhBase + l] >> 4) & 3) << 4)) - 32;
                            var q4 = ((b[qlBase + l + 32] >> 4) | (((b[qhBase + l] >> 6) & 3) << 4)) - 32;

                            var sc0 = (sbyte)b[scBase + isIdx + 0];
                            var sc2 = (sbyte)b[scBase + isIdx + 2];
                            var sc4 = (sbyte)b[scBase + isIdx + 4];
                            var sc6 = (sbyte)b[scBase + isIdx + 6];

                            deqBuf[dstBase + l] = d * sc0 * q1;
                            deqBuf[dstBase + l + 32] = d * sc2 * q2;
                            deqBuf[dstBase + l + 64] = d * sc4 * q3;
                            deqBuf[dstBase + l + 96] = d * sc6 * q4;
                        }
                    }
                }

                result[c] = DotProduct(a, deqBuf);
            }
        }
        finally
        {
            if (rentedBuf is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rentedBuf);
        }
    }

    #endregion

    #region Q2_K/Q3_K/Q4_K/Q5_K Fused MatMul (New)

    private static void MatMulFusedQ2_K(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int qk = 256;
        const int blockSizeBytes = 84;
        var numBlocks = aCols / qk;

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockSizeBytes;
            var dot = 0f;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockSizeBytes);
                var d = HalfHelper.HalfToFloat(b, off);
                var dmin = HalfHelper.HalfToFloat(b, off + 2);
                var baseIdx = blk * qk;

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 64;
                    var local = i % 64;
                    var sub = local / 16;

                    var scIdx = g * 4 + sub;
                    var sc = b[off + 4 + scIdx * 2] & 0x0F;
                    var mi = b[off + 4 + scIdx * 2 + 1] & 0x0F;

                    var q = (b[off + 20 + i / 4] >> ((i % 4) * 2)) & 0x3;

                    dot += a[baseIdx + i] * (d * sc * (q - 0.5f) - dmin * mi);
                }
            }

            result[c] = dot;
        }
    }

    private static void MatMulFusedQ3_K(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int qk = 256;
        const int blockSizeBytes = 110;
        var numBlocks = aCols / qk;

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockSizeBytes;
            var dot = 0f;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockSizeBytes);
                var d = HalfHelper.HalfToFloat(b, off);
                var baseIdx = blk * qk;

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 16;
                    var sc = Extract6Bit(b, off + 4, g * 6);
                    var q = (b[off + 48 + i / 4] >> ((i % 4) * 2)) & 0x3;
                    var m = (b[off + 16 + i / 8] >> (i % 8)) & 0x1;
                    var value = q - (m << 2);

                    dot += a[baseIdx + i] * d * sc * value;
                }
            }

            result[c] = dot;
        }
    }

    private static void MatMulFusedQ4_K(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int qk = 256;
        const int blockSizeBytes = 144;
        var numBlocks = aCols / qk;

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockSizeBytes;
            var dot = 0f;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockSizeBytes);
                var d = HalfHelper.HalfToFloat(b, off);
                var dmin = HalfHelper.HalfToFloat(b, off + 2);
                var baseIdx = blk * qk;

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 32;
                    var sc = Extract6Bit(b, off + 4, g * 12);
                    var mi = Extract6Bit(b, off + 4, g * 12 + 6);
                    var q = (b[off + 16 + i / 2] >> ((i % 2) * 4)) & 0xF;

                    dot += a[baseIdx + i] * (d * sc * (q - 8) - dmin * mi);
                }
            }

            result[c] = dot;
        }
    }

    private static void MatMulFusedQ5_K(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, int aCols, int bCols)
    {
        const int qk = 256;
        const int blockSizeBytes = 176;
        var numBlocks = aCols / qk;

        for (var c = 0; c < bCols; c++)
        {
            var rowOffset = (long)c * numBlocks * blockSizeBytes;
            var dot = 0f;

            for (var blk = 0; blk < numBlocks; blk++)
            {
                var off = (int)(rowOffset + blk * blockSizeBytes);
                var d = HalfHelper.HalfToFloat(b, off);
                var dmin = HalfHelper.HalfToFloat(b, off + 2);
                var baseIdx = blk * qk;

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 32;
                    var sc = Extract6Bit(b, off + 4, g * 12);
                    var mi = Extract6Bit(b, off + 4, g * 12 + 6);

                    var qLow = (b[off + 48 + i / 2] >> ((i % 2) * 4)) & 0xF;
                    var qHigh = (b[off + 16 + i / 8] >> (i % 8)) & 0x1;
                    var q5 = qLow + (qHigh << 4);

                    dot += a[baseIdx + i] * (d * sc * (q5 - 16) - dmin * mi);
                }
            }

            result[c] = dot;
        }
    }

    #endregion

    #region F32 MatMul (Optimized with better tiling for single-row)

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
                var bRow = b.Slice(c * aCols, aCols);
                var accVec = Vector<float>.Zero;
                var k = 0;

                for (; k <= aCols - vecSize; k += vecSize)
                {
                    var va = new Vector<float>(aRow.Slice(k, vecSize));
                    var vb = new Vector<float>(bRow.Slice(k, vecSize));
                    accVec += va * vb;
                }

                var dot = Vector.Dot(accVec, Vector<float>.One);

                for (; k < aCols; k++)
                    dot += aRow[k] * bRow[k];

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
                var bRow = b.Slice(c * aCols, aCols);
                var accVec = Vector<float>.Zero;
                var k = 0;

                for (; k <= aCols - vecSize; k += vecSize)
                {
                    var va = new Vector<float>(a.Slice(k, vecSize));
                    var vb = new Vector<float>(bRow.Slice(k, vecSize));
                    accVec += va * vb;
                }

                var dot = Vector.Dot(accVec, Vector<float>.One);
                for (; k < aCols; k++)
                    dot += a[k] * bRow[k];

                result[c] = dot;
            }
        }
    }

    #endregion

    #region Utility Methods

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

    #endregion

    #region Dequantize Row

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
            case GgmlType.Q2_K:
                DequantizeK.DequantizeQ2_K(slice, dst, 1, rowElements);
                break;
            case GgmlType.Q3_K:
                DequantizeK.DequantizeQ3_K(slice, dst, 1, rowElements);
                break;
            case GgmlType.Q4_K:
                DequantizeK.DequantizeQ4_K(slice, dst, 1, rowElements);
                break;
            case GgmlType.Q5_K:
                DequantizeK.DequantizeQ5_K(slice, dst, 1, rowElements);
                break;
            case GgmlType.Q6_K:
                DequantizeK.DequantizeQ6_K(slice, dst, 1, rowElements);
                break;
            default:
                DequantizeGeneric(slice, dst, type, 1, rowElements);
                break;
        }
    }

    #endregion

    #region DequantizeGeneric

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
                        for (var j = 0; j < blockElements / 2; j++)
                        {
                            var qs = src[off + 4 + j];
                            dst[r * rowElements + b * blockElements + j] = d * (qs & 0x0F) + m;
                            dst[r * rowElements + b * blockElements + j + blockElements / 2] = d * ((qs >> 4) & 0x0F) + m;
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
                        var qh = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(off + 2, 4));
                        for (var j = 0; j < blockElements / 2; j++)
                        {
                            var qs = src[off + 6 + j];
                            var x0 = ((qs & 0x0F) | (int)(((qh >> (j + 0)) << 4) & 0x10)) - 16;
                            var x1 = (((qs >> 4) & 0x0F) | (int)((qh >> (j + 12)) & 0x10)) - 16;
                            dst[r * rowElements + b * blockElements + j] = x0 * d;
                            dst[r * rowElements + b * blockElements + j + blockElements / 2] = x1 * d;
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
                        var qh = BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(off + 4, 4));
                        for (var j = 0; j < blockElements / 2; j++)
                        {
                            var qs = src[off + 8 + j];
                            var x0 = (qs & 0x0F) | (int)(((qh >> (j + 0)) << 4) & 0x10);
                            var x1 = ((qs >> 4) & 0x0F) | (int)((qh >> (j + 12)) & 0x10);
                            dst[r * rowElements + b * blockElements + j] = d * x0 + m;
                            dst[r * rowElements + b * blockElements + j + blockElements / 2] = d * x1 + m;
                        }
                    }
                    break;
                }
            default:
                throw new NotSupportedException($"Dequantization for type {type} not yet implemented.");
        }
    }

    #endregion

    #region RoPE (P7: Pre-computed cos/sin tables with SIMD)

    public static void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim)
    {
        var actualRotaryDim = rotaryDim > 0 ? rotaryDim : headDim;
        var halfDim = actualRotaryDim / 2;
        var sameSpan = query == key;
        var vecSize = Vector<float>.Count;

        Span<float> cosBuf = halfDim <= 256 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinBuf = halfDim <= 256 ? stackalloc float[halfDim] : new float[halfDim];

        for (var i = 0; i < halfDim; i++)
        {
            var freq = 1f / MathF.Pow(freqBase, 2f * i / actualRotaryDim);
            var angle = position * freq;
            cosBuf[i] = MathF.Cos(angle);
            sinBuf[i] = MathF.Sin(angle);
        }

        ApplyRoPECore(query, key, cosBuf, sinBuf, halfDim, sameSpan, vecSize);
    }

    public static void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, ReadOnlySpan<float> freqTable)
    {
        var halfDim = freqTable.Length;
        var sameSpan = query == key;
        var vecSize = Vector<float>.Count;

        Span<float> cosBuf = halfDim <= 256 ? stackalloc float[halfDim] : new float[halfDim];
        Span<float> sinBuf = halfDim <= 256 ? stackalloc float[halfDim] : new float[halfDim];

        for (var i = 0; i < halfDim; i++)
        {
            var angle = position * freqTable[i];
            cosBuf[i] = MathF.Cos(angle);
            sinBuf[i] = MathF.Sin(angle);
        }

        ApplyRoPECore(query, key, cosBuf, sinBuf, halfDim, sameSpan, vecSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRoPECore(Span<float> query, Span<float> key, ReadOnlySpan<float> cosBuf, ReadOnlySpan<float> sinBuf, int halfDim, bool sameSpan, int vecSize)
    {
        var i = 0;
        for (; i <= halfDim - vecSize; i += vecSize)
        {
            var cosVec = new Vector<float>(cosBuf.Slice(i, vecSize));
            var sinVec = new Vector<float>(sinBuf.Slice(i, vecSize));

            var q0Vec = new Vector<float>(query.Slice(i, vecSize));
            var q1Vec = new Vector<float>(query.Slice(i + halfDim, vecSize));
            var rotQ0 = q0Vec * cosVec - q1Vec * sinVec;
            var rotQ1 = q0Vec * sinVec + q1Vec * cosVec;
            rotQ0.CopyTo(query.Slice(i, vecSize));
            rotQ1.CopyTo(query.Slice(i + halfDim, vecSize));

            if (sameSpan) continue;

            var k0Vec = new Vector<float>(key.Slice(i, vecSize));
            var k1Vec = new Vector<float>(key.Slice(i + halfDim, vecSize));
            var rotK0 = k0Vec * cosVec - k1Vec * sinVec;
            var rotK1 = k0Vec * sinVec + k1Vec * cosVec;
            rotK0.CopyTo(key.Slice(i, vecSize));
            rotK1.CopyTo(key.Slice(i + halfDim, vecSize));
        }

        for (; i < halfDim; i++)
        {
            var cosVal = cosBuf[i];
            var sinVal = sinBuf[i];
            var idx = i;
            var halfIdx = i + halfDim;

            var q0 = query[idx];
            var q1 = query[halfIdx];
            query[idx] = q0 * cosVal - q1 * sinVal;
            query[halfIdx] = q0 * sinVal + q1 * cosVal;

            if (sameSpan) continue;

            var k0 = key[idx];
            var k1 = key[halfIdx];
            key[idx] = k0 * cosVal - k1 * sinVal;
            key[halfIdx] = k0 * sinVal + k1 * cosVal;
        }
    }

    public static void Conv1D(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int kernelSize, int inputDim)
    {
        if (kernelSize == 1)
        {
            var vecSize = Vector<float>.Count;
            var i = 0;
            for (; i <= inputDim - vecSize; i += vecSize)
            {
                var wVec = new Vector<float>(weights.Slice(i, vecSize));
                var iVec = new Vector<float>(input.Slice(i, vecSize));
                (iVec * wVec).CopyTo(output.Slice(i, vecSize));
            }
            for (; i < inputDim; i++)
                output[i] = input[i] * weights[i];
        }
        else
        {
            var vecSize = Vector<float>.Count;
            for (var c = 0; c < inputDim; c++)
            {
                var sumVec = Vector<float>.Zero;
                var k = 0;
                for (; k <= kernelSize - vecSize; k += vecSize)
                {
                    var iv = new Vector<float>(input.Slice(c * kernelSize + k, vecSize));
                    var wv = new Vector<float>(weights.Slice(c * kernelSize + k, vecSize));
                    sumVec += iv * wv;
                }
                var sum = Vector.Dot(sumVec, Vector<float>.One);
                for (; k < kernelSize; k++)
                    sum += input[c * kernelSize + k] * weights[c * kernelSize + k];
                output[c] = sum;
            }
        }
    }

    private static int Extract6Bit(ReadOnlySpan<byte> src, int baseOffset, int bitOffset)
        => Dequantize.DequantizeK.Extract6Bit(src, baseOffset, bitOffset);

    #endregion
}