using System.Runtime.InteropServices;

namespace Dotllm.Tensors.Numeric;

internal static class HalfHelper
{
    public static float HalfToFloat(ushort half)
    {
        var sign = (half >> 15) & 0x1;
        var exponent = (half >> 10) & 0x1F;
        var mantissa = half & 0x3FF;

        if (exponent == 0)
        {
            if (mantissa == 0)
                return sign != 0 ? -0f : 0f;
            var normalized = mantissa;
            var shift = 0;
            while ((normalized & 0x400) == 0) { normalized <<= 1; shift++; }
            var fSign = sign != 0 ? -1f : 1f;
            return fSign * (normalized & 0x3FF) / 1024f * MathF.Pow(2f, 1 - shift - 15);
        }

        if (exponent == 31)
            return mantissa == 0 ? (sign != 0 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;

        var fSign2 = sign != 0 ? -1f : 1f;
        return fSign2 * (1f + mantissa / 1024f) * MathF.Pow(2f, exponent - 15);
    }

    public static float HalfToFloat(ReadOnlySpan<byte> src, int offset)
    {
        var half = (ushort)(src[offset] | (src[offset + 1] << 8));
        return HalfToFloat(half);
    }
}