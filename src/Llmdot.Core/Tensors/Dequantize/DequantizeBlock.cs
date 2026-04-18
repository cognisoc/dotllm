using System.Numerics;
using System.Runtime.InteropServices;
using Llmdot.Tensors.Numeric;

namespace Llmdot.Tensors.Dequantize;

internal static class DequantizeQ4_0
{
    public static void Dequantize(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        var blockElements = 32;
        var blockSizeBytes = 18;
        var blocksPerRow = rowElements / blockElements;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var blockOffset = rowOffset + b * blockSizeBytes;
                var scale = HalfHelper.HalfToFloat(src, blockOffset);

                for (var j = 0; j < blockElements / 2; j++)
                {
                    var qs = src[blockOffset + 2 + j];
                    dst[dstOffset + b * blockElements + j] = ((qs & 0x0F) - 8) * scale;
                    dst[dstOffset + b * blockElements + j + blockElements / 2] = (((qs >> 4) & 0x0F) - 8) * scale;
                }
            }
        }
    }
}

internal static class DequantizeQ8_0
{
    public static void Dequantize(ReadOnlySpan<byte> src, Span<float> dst, int numRows, int rowElements)
    {
        var blockElements = 32;
        var blockSizeBytes = 34;
        var blocksPerRow = rowElements / blockElements;

        for (var r = 0; r < numRows; r++)
        {
            var rowOffset = r * blocksPerRow * blockSizeBytes;
            var dstOffset = r * rowElements;

            for (var b = 0; b < blocksPerRow; b++)
            {
                var blockOffset = rowOffset + b * blockSizeBytes;
                var scale = HalfHelper.HalfToFloat(src, blockOffset);

                for (var i = 0; i < blockElements; i++)
                    dst[dstOffset + b * blockElements + i] = (sbyte)src[blockOffset + 2 + i] * scale;
            }
        }
    }
}

internal static class DequantizeF16
{
    public static void Dequantize(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        for (var i = 0; i < count; i++)
            dst[i] = HalfHelper.HalfToFloat(src, i * 2);
    }
}

internal static class DequantizeBF16
{
    public static void Dequantize(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        var vecSize = Vector<float>.Count;
        var i = 0;

        for (; i <= count - vecSize; i += vecSize)
        {
            for (var v = 0; v < vecSize; v++)
            {
                var hiByte = src[(i + v) * 2 + 1];
                var loByte = src[(i + v) * 2];
                var bits = (hiByte << 24) | (loByte << 16);
                dst[i + v] = BitConverter.Int32BitsToSingle(bits);
            }
        }

        for (; i < count; i++)
        {
            var b0 = src[i * 2];
            var b1 = src[i * 2 + 1];
            var bits = (uint)((b1 << 24) | (b0 << 16));
            dst[i] = BitConverter.Int32BitsToSingle((int)bits);
        }
    }
}

internal static class DequantizeF32
{
    public static void Dequantize(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        var srcFloats = MemoryMarshal.Cast<byte, float>(src);
        srcFloats[..count].CopyTo(dst);
    }
}