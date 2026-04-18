using System.Runtime.InteropServices;
using Llmdot.Tensors.Numeric;

namespace Llmdot.Tensors.Dequantize;

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

                for (var n = 0; n < qk; n += 128)
                {
                    var qlBase = off + n / 2;
                    var qhBase = off + 128 + n / 4;
                    var scBase = off + 192 + (n / 16);

                    for (var l = 0; l < 32; l++)
                    {
                        var isIdx = l / 16;

                        var q1 = ((src[qlBase + l] & 0xF) | (((src[qhBase + l] >> 0) & 3) << 4)) - 32;
                        var q2 = ((src[qlBase + l + 32] & 0xF) | (((src[qhBase + l] >> 2) & 3) << 4)) - 32;
                        var q3 = ((src[qlBase + l] >> 4) | (((src[qhBase + l] >> 4) & 3) << 4)) - 32;
                        var q4 = ((src[qlBase + l + 32] >> 4) | (((src[qhBase + l] >> 6) & 3) << 4)) - 32;

                        var sc0 = (sbyte)src[scBase + isIdx + 0];
                        var sc2 = (sbyte)src[scBase + isIdx + 2];
                        var sc4 = (sbyte)src[scBase + isIdx + 4];
                        var sc6 = (sbyte)src[scBase + isIdx + 6];

                        var dstBase = dstOffset + b * qk + n;
                        dst[dstBase + l] = d * sc0 * q1;
                        dst[dstBase + l + 32] = d * sc2 * q2;
                        dst[dstBase + l + 64] = d * sc4 * q3;
                        dst[dstBase + l + 96] = d * sc6 * q4;
                    }
                }
            }
        }
    }

    internal static int Extract6Bit(ReadOnlySpan<byte> src, int baseOffset, int bitOffset)
    {
        var byteIdx = bitOffset / 8;
        var bitShift = bitOffset % 8;

        if (bitShift <= 2)
            return (src[baseOffset + byteIdx] >> bitShift) & 0x3F;

        return ((src[baseOffset + byteIdx] >> bitShift) |
                (src[baseOffset + byteIdx + 1] << (8 - bitShift))) & 0x3F;
    }
}