using System.Numerics;
using System.Runtime.CompilerServices;

namespace Dotllm.Tensors.Numeric;

internal static class VectorMath
{
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

    public static void Silu(ReadOnlySpan<float> input, Span<float> result)
    {
        for (var i = 0; i < input.Length; i++)
        {
            var x = input[i];
            result[i] = x / (1f + MathF.Exp(-x));
        }
    }

    public static void SiluInPlace(Span<float> input)
    {
        for (var i = 0; i < input.Length; i++)
        {
            var x = input[i];
            input[i] = x / (1f + MathF.Exp(-x));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x) =>
        0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));

    public static void Gelu(ReadOnlySpan<float> input, Span<float> result)
    {
        for (var i = 0; i < input.Length; i++)
            result[i] = Gelu(input[i]);
    }

    public static void Softmax(Span<float> input, float? softcap = null)
    {
        if (input.Length == 0) return;

        if (softcap.HasValue)
        {
            var cap = softcap.Value;
            for (var i = 0; i < input.Length; i++)
                input[i] = cap * MathF.Tanh(input[i] / cap);
        }

        var max = input[0];
        for (var i = 1; i < input.Length; i++)
            if (input[i] > max) max = input[i];

        var sum = 0f;
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = MathF.Exp(input[i] - max);
            sum += input[i];
        }

        var invSum = 1f / sum;
        Scale(input, invSum);
    }

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
        for (; i <= n - vecSize; i += vecSize)
        {
            var vIn = new Vector<float>(input.Slice(i, vecSize));
            var vW = new Vector<float>(weights.Slice(i, vecSize));
            var vB = new Vector<float>(bias.Slice(i, vecSize));
            (vW * (vIn - vMean) * vInvStd + vB).CopyTo(output.Slice(i, vecSize));
        }

        for (; i < n; i++)
            output[i] = weights[i] * (input[i] - mean) * invStd + bias[i];
    }

    public static void Softcap(Span<float> input, float cap)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = cap * MathF.Tanh(input[i] / cap);
    }

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
}