using System.Numerics;
using System.Runtime.CompilerServices;

namespace Llmdot.Tensors.Numeric;

internal static class VectorMath
{
    #region Vectorized Arithmetic

    public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        var n = result.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        for (; i <= n - vecSize; i += vecSize)
        {
            var va = new Vector<float>(a.Slice(i, vecSize));
            var vb = new Vector<float>(b.Slice(i, vecSize));
            (va + vb).CopyTo(result.Slice(i, vecSize));
        }

        for (; i < n; i++)
            result[i] = a[i] + b[i];
    }

    public static void Add(Span<float> a, ReadOnlySpan<float> b)
    {
        var n = a.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        for (; i <= n - vecSize; i += vecSize)
        {
            var va = new Vector<float>(a.Slice(i, vecSize));
            var vb = new Vector<float>(b.Slice(i, vecSize));
            (va + vb).CopyTo(a.Slice(i, vecSize));
        }

        for (; i < n; i++)
            a[i] += b[i];
    }

    public static void Scale(ReadOnlySpan<float> input, float scale, Span<float> result)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;
        var vs = new Vector<float>(scale);

        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            (v * vs).CopyTo(result.Slice(i, vecSize));
        }

        for (; i < n; i++)
            result[i] = input[i] * scale;
    }

    public static void Scale(Span<float> input, float scale)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;
        var vs = new Vector<float>(scale);

        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            (v * vs).CopyTo(input.Slice(i, vecSize));
        }

        for (; i < n; i++)
            input[i] *= scale;
    }

    public static void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        var n = result.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        for (; i <= n - vecSize; i += vecSize)
        {
            var va = new Vector<float>(a.Slice(i, vecSize));
            var vb = new Vector<float>(b.Slice(i, vecSize));
            (va * vb).CopyTo(result.Slice(i, vecSize));
        }

        for (; i < n; i++)
            result[i] = a[i] * b[i];
    }

    #endregion

    #region Vectorized Activations (P4)

    public static void Silu(ReadOnlySpan<float> input, Span<float> result)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        for (; i <= n - vecSize; i += vecSize)
        {
            var vx = new Vector<float>(input.Slice(i, vecSize));
            (vx / (Vector<float>.One + ExpApproxVector(-vx))).CopyTo(result.Slice(i, vecSize));
        }

        for (; i < n; i++)
        {
            var x = input[i];
            result[i] = x / (1f + MathF.Exp(-x));
        }
    }

    public static void SiluInPlace(Span<float> input)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        for (; i <= n - vecSize; i += vecSize)
        {
            var vx = new Vector<float>(input.Slice(i, vecSize));
            (vx / (Vector<float>.One + ExpApproxVector(-vx))).CopyTo(input.Slice(i, vecSize));
        }

        for (; i < n; i++)
        {
            var x = input[i];
            input[i] = x / (1f + MathF.Exp(-x));
        }
    }

    private static Vector<float> ExpApproxVector(Vector<float> x)
    {
        var vLog2E = new Vector<float>(1.4426950408889634f);
        var vHalf = new Vector<float>(0.5f);
        var vOne = Vector<float>.One;
        var vLn2Hi = new Vector<float>(0.693145751953125f);
        var vLn2Lo = new Vector<float>(1.42860677e-6f);

        var clamped = Vector.Min(Vector.Max(x, new Vector<float>(-87.3f)), new Vector<float>(88.4f));

        var k = Vector.ConvertToInt32(clamped * vLog2E + vHalf);

        var kFlt = Vector.ConvertToSingle(k);
        var y = clamped - kFlt * vLn2Hi - kFlt * vLn2Lo;
        var c = y * y * vHalf;
        var pade = vOne + y + c + c * y * new Vector<float>(0.3333333f) + c * y * y * new Vector<float>(0.2f);

        var v127 = new Vector<int>(127);
        var scaleBitPattern = Vector.ShiftLeft(k + v127, 23);
        var scale = Vector.AsVectorSingle(scaleBitPattern);

        var result = pade * scale;

        var underflow = Vector.LessThan(x, new Vector<float>(-87.3f));
        var overflow = Vector.GreaterThan(x, new Vector<float>(88.4f));
        result = Vector.ConditionalSelect(underflow, Vector<float>.Zero, result);
        result = Vector.ConditionalSelect(overflow, new Vector<float>(float.PositiveInfinity), result);

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x) =>
        0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

    public static void Gelu(ReadOnlySpan<float> input, Span<float> result)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;
        var v05 = new Vector<float>(0.5f);
        var vOne = Vector<float>.One;
        var vCoeff = new Vector<float>(0.7978845608028654f);
        var vCoeff3 = new Vector<float>(0.044715f);

        for (; i <= n - vecSize; i += vecSize)
        {
            var vx = new Vector<float>(input.Slice(i, vecSize));
            var vTanh = TanhApproxVector(vx * vCoeff * (vOne + vCoeff3 * vx * vx * vx));
            (v05 * vx * (vOne + vTanh)).CopyTo(result.Slice(i, vecSize));
        }

        for (; i < n; i++)
        {
            var x = input[i];
            result[i] = 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        }
    }

    private static Vector<float> TanhApproxVector(Vector<float> x)
    {
        var v27 = new Vector<float>(27f);
        var v9 = new Vector<float>(9f);
        var vOne = Vector<float>.One;
        var vNegOne = new Vector<float>(-1f);

        var x2 = x * x;
        var approx = x * (v27 + x2) / (v27 + v9 * x2);

        var large = Vector.GreaterThan(Vector.Abs(x), new Vector<float>(4f));
        var sign = Vector.ConditionalSelect(Vector.GreaterThan(x, Vector<float>.Zero), vOne, vNegOne);
        return Vector.ConditionalSelect(large, sign, approx);
    }

    #endregion

    #region Vectorized Softmax (P5)

    public static void Softmax(Span<float> input, float? softcap = null)
    {
        if (input.Length == 0) return;

        var n = input.Length;
        var vecSize = Vector<float>.Count;

        if (softcap.HasValue)
        {
            var cap = softcap.Value;
            var vCap = new Vector<float>(cap);
            var vInvCap = new Vector<float>(1f / cap);
            var si = 0;
            for (; si <= n - vecSize; si += vecSize)
            {
                var v = new Vector<float>(input.Slice(si, vecSize));
                (vCap * TanhApproxVector(v * vInvCap)).CopyTo(input.Slice(si, vecSize));
            }
            for (; si < n; si++)
                input[si] = cap * MathF.Tanh(input[si] / cap);
        }

        var max = FindMax(input, n, vecSize);
        var vMaxVal = new Vector<float>(max);
        var vSumVec = Vector<float>.Zero;

        var j = 0;
        for (; j <= n - vecSize; j += vecSize)
        {
            var v = ExpApproxVector(new Vector<float>(input.Slice(j, vecSize)) - vMaxVal);
            v.CopyTo(input.Slice(j, vecSize));
            vSumVec += v;
        }

        var sum = Vector.Dot(vSumVec, Vector<float>.One);
        for (; j < n; j++)
        {
            input[j] = MathF.Exp(input[j] - max);
            sum += input[j];
        }

        var invSum = 1f / sum;
        Scale(input, invSum);
    }

    private static float FindMax(Span<float> input, int n, int vecSize)
    {
        if (n >= vecSize)
        {
            var vMaxLocal = new Vector<float>(input.Slice(0, vecSize));
            var mi = vecSize;
            for (; mi <= n - vecSize; mi += vecSize)
            {
                var v = new Vector<float>(input.Slice(mi, vecSize));
                vMaxLocal = Vector.Max(vMaxLocal, v);
            }

            var max = float.NegativeInfinity;
            for (var k = 0; k < vecSize; k++)
                if (vMaxLocal[k] > max) max = vMaxLocal[k];
            for (; mi < n; mi++)
                if (input[mi] > max) max = input[mi];
            return max;
        }

        var maxSmall = input[0];
        for (var mi = 1; mi < n; mi++)
            if (input[mi] > maxSmall) maxSmall = input[mi];
        return maxSmall;
    }

    public static void Softcap(Span<float> input, float cap)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;
        var vCap = new Vector<float>(cap);
        var vInvCap = new Vector<float>(1f / cap);

        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            (vCap * TanhApproxVector(v * vInvCap)).CopyTo(input.Slice(i, vecSize));
        }

        for (; i < n; i++)
            input[i] = cap * MathF.Tanh(input[i] / cap);
    }

    #endregion

    #region Vectorized Normalization

    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;
        var ssVec = Vector<float>.Zero;

        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            ssVec += v * v;
        }

        var ss = Vector.Dot(ssVec, Vector<float>.One);
        for (; i < n; i++)
            ss += input[i] * input[i];

        ss /= n;
        ss += epsilon;
        var invNorm = 1f / MathF.Sqrt(ss);
        var vInvNorm = new Vector<float>(invNorm);

        i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            var vIn = new Vector<float>(input.Slice(i, vecSize));
            var vW = new Vector<float>(weights.Slice(i, vecSize));
            (vW * vIn * vInvNorm).CopyTo(output.Slice(i, vecSize));
        }

        for (; i < n; i++)
            output[i] = weights[i] * (input[i] * invNorm);
    }

    public static void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon)
    {
        var n = input.Length;
        var i = 0;
        var vecSize = Vector<float>.Count;

        var sumVec = Vector<float>.Zero;
        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            sumVec += v;
        }
        var mean = Vector.Dot(sumVec, Vector<float>.One);
        for (; i < n; i++)
            mean += input[i];
        mean /= n;

        var vMean = new Vector<float>(mean);
        i = 0;
        var varVec = Vector<float>.Zero;
        for (; i <= n - vecSize; i += vecSize)
        {
            var v = new Vector<float>(input.Slice(i, vecSize));
            var d = v - vMean;
            varVec += d * d;
        }
        var variance = Vector.Dot(varVec, Vector<float>.One);
        for (; i < n; i++)
        {
            var d = input[i] - mean;
            variance += d * d;
        }
        variance /= n;

        var invStd = 1f / MathF.Sqrt(variance + epsilon);
        var vInvStd = new Vector<float>(invStd);

        i = 0;
        var hasBias = bias.Length > 0;
        if (hasBias)
        {
            for (; i <= n - vecSize; i += vecSize)
            {
                var vIn = new Vector<float>(input.Slice(i, vecSize));
                var vW = new Vector<float>(weights.Slice(i, vecSize));
                var vB = new Vector<float>(bias.Slice(i, vecSize));
                (vW * (vIn - vMean) * vInvStd + vB).CopyTo(output.Slice(i, vecSize));
            }
        }
        else
        {
            for (; i <= n - vecSize; i += vecSize)
            {
                var vIn = new Vector<float>(input.Slice(i, vecSize));
                var vW = new Vector<float>(weights.Slice(i, vecSize));
                (vW * (vIn - vMean) * vInvStd).CopyTo(output.Slice(i, vecSize));
            }
        }

        for (; i < n; i++)
        {
            var val = weights[i] * (input[i] - mean) * invStd;
            output[i] = hasBias ? val + bias[i] : val;
        }
    }

    #endregion

    #region ArgMax

    public static int ArgMax(ReadOnlySpan<float> input)
    {
        var maxIdx = 0;
        var maxVal = input[0];
        for (var i = 1; i < input.Length; i++)
        {
            if (input[i] > maxVal)
            {
                maxVal = input[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    #endregion
}