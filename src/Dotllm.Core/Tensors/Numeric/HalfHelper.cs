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

            var newExp = 127 - 15 - shift;
            var newMantissa = (normalized & 0x3FF) << 13;
            var bits = ((uint)sign << 31) | ((uint)newExp << 23) | (uint)newMantissa;
            return BitConverter.Int32BitsToSingle((int)bits);
        }

        if (exponent == 31)
            return mantissa == 0 ? (sign != 0 ? float.NegativeInfinity : float.PositiveInfinity) : float.NaN;

        var floatExp = exponent - 15 + 127;
        var floatBits = ((uint)sign << 31) | ((uint)floatExp << 23) | ((uint)mantissa << 13);
        return BitConverter.Int32BitsToSingle((int)floatBits);
    }

    public static float HalfToFloat(ReadOnlySpan<byte> src, int offset)
    {
        var half = (ushort)(src[offset] | (src[offset + 1] << 8));
        return HalfToFloat(half);
    }
}