using System.Runtime.InteropServices;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Tensors.Dequantize;

internal static class DequantizeK
{
    public static void DequantizeQ2_K(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        const int qk = 256;
        const int blockSizeBytes = 84;
        var blocksPerRow = rowElements / qk;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var off = rowOffset + b * blockSizeBytes;
                var d = HalfHelper.HalfToFloat(src, off);
                var dmin = HalfHelper.HalfToFloat(src, off + 2);

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 64;
                    var local = i % 64;
                    var sub = local / 16;

                    var scIdx = g * 4 + sub;
                    var sc = (src[off + 4 + scIdx * 2] & 0x0F);
                    var mi = (src[off + 4 + scIdx * 2 + 1] & 0x0F);

                    var q = (src[off + 20 + i / 4] >> ((i % 4) * 2)) & 0x3;

                    dst[dstOffset + b * qk + i] = d * sc * (q - 0.5f) - dmin * mi;
                }
            }
        }
    }

    public static void DequantizeQ3_K(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        const int qk = 256;
        const int blockSizeBytes = 110;
        var blocksPerRow = rowElements / qk;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var off = rowOffset + b * blockSizeBytes;
                var d = HalfHelper.HalfToFloat(src, off);

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 16;

                    var sc = Extract6Bit(src, off + 4, g * 6);

                    var q = (src[off + 48 + i / 4] >> ((i % 4) * 2)) & 0x3;
                    var m = (src[off + 16 + i / 8] >> (i % 8)) & 0x1;

                    var value = q - (m << 2);

                    dst[dstOffset + b * qk + i] = d * sc * value;
                }
            }
        }
    }

    public static void DequantizeQ4_K(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        const int qk = 256;
        const int blockSizeBytes = 144;
        var blocksPerRow = rowElements / qk;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var off = rowOffset + b * blockSizeBytes;
                var d = HalfHelper.HalfToFloat(src, off);
                var dmin = HalfHelper.HalfToFloat(src, off + 2);

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 32;

                    var sc = Extract6Bit(src, off + 4, g * 12);
                    var mi = Extract6Bit(src, off + 4, g * 12 + 6);

                    var q = (src[off + 16 + i / 2] >> ((i % 2) * 4)) & 0xF;

                    dst[dstOffset + b * qk + i] = d * sc * (q - 8) - dmin * mi;
                }
            }
        }
    }

    public static void DequantizeQ5_K(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        const int qk = 256;
        const int blockSizeBytes = 176;
        var blocksPerRow = rowElements / qk;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var off = rowOffset + b * blockSizeBytes;
                var d = HalfHelper.HalfToFloat(src, off);
                var dmin = HalfHelper.HalfToFloat(src, off + 2);

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 32;

                    var sc = Extract6Bit(src, off + 4, g * 12);
                    var mi = Extract6Bit(src, off + 4, g * 12 + 6);

                    var qLow = (src[off + 48 + i / 2] >> ((i % 2) * 4)) & 0xF;
                    var qHigh = (src[off + 16 + i / 8] >> (i % 8)) & 0x1;
                    var q5 = qLow + (qHigh << 4);

                    dst[dstOffset + b * qk + i] = d * sc * (q5 - 16) - dmin * mi;
                }
            }
        }
    }

    public static void DequantizeQ6_K(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        const int qk = 256;
        const int blockSizeBytes = 210;
        var blocksPerRow = rowElements / qk;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var off = rowOffset + b * blockSizeBytes;
                var d = HalfHelper.HalfToFloat(src, off + 208);

                for (var i = 0; i < qk; i++)
                {
                    var g = i / 16;
                    var j = i % 16;

                    int qlVal;
                    if (g < 8)
                        qlVal = src[off + 16 * g + j];
                    else
                        qlVal = src[off + 16 * (g - 8) + j + 128];

                    var qhIdx = 4 * g + j / 4;
                    var qhShift = 2 * (j % 4);
                    var qhBits = (src[off + 128 + qhIdx] >> qhShift) & 0x3;

                    var q6 = (qlVal | (qhBits << 4)) - 32;
                    var sc = (sbyte)src[off + 192 + g];

                    dst[dstOffset + b * qk + i] = d * sc * q6;
                }
            }
        }
    }

    private static int Extract6Bit(ReadOnlySpan<byte> src, int baseOffset, int bitOffset)
    {
        var byteIdx = bitOffset / 8;
        var bitShift = bitOffset % 8;

        if (bitShift <= 2)
            return (src[baseOffset + byteIdx] >> bitShift) & 0x3F;

        return ((src[baseOffset + byteIdx] >> bitShift) |
                (src[baseOffset + byteIdx + 1] << (8 - bitShift))) & 0x3F;
    }
}