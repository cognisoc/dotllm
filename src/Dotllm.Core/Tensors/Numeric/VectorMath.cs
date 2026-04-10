using System.Runtime.InteropServices;

namespace Dotllm.Tensors.Numeric;

internal static class VectorMath
{
    public static void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        for (var i = 0; i < result.Length; i++)
            result[i] = a[i] + b[i];
    }

    public static void Add(Span<float> a, ReadOnlySpan<float> b)
    {
        for (var i = 0; i < a.Length; i++)
            a[i] += b[i];
    }

    public static void Scale(ReadOnlySpan<float> input, float scale, Span<float> result)
    {
        for (var i = 0; i < input.Length; i++)
            result[i] = input[i] * scale;
    }

    public static void Scale(Span<float> input, float scale)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] *= scale;
    }

    public static void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        for (var i = 0; i < result.Length; i++)
            result[i] = a[i] * b[i];
    }

    public static void Silu(ReadOnlySpan<float> input, Span<float> result)
    {
        for (var i = 0; i < input.Length; i++)
            result[i] = input[i] / (1f + MathF.Exp(-input[i]));
    }

    public static void SiluInPlace(Span<float> input)
    {
        for (var i = 0; i < input.Length; i++)
            input[i] = input[i] / (1f + MathF.Exp(-input[i]));
    }

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
        for (var i = 0; i < input.Length; i++)
            input[i] *= invSum;
    }

    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon)
    {
        var ss = 0f;
        for (var i = 0; i < input.Length; i++)
            ss += input[i] * input[i];
        ss /= input.Length;
        ss += epsilon;
        var invNorm = 1f / MathF.Sqrt(ss);

        for (var i = 0; i < input.Length; i++)
            output[i] = weights[i] * (input[i] * invNorm);
    }

    public static void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon)
    {
        var mean = 0f;
        for (var i = 0; i < input.Length; i++)
            mean += input[i];
        mean /= input.Length;

        var variance = 0f;
        for (var i = 0; i < input.Length; i++)
        {
            var d = input[i] - mean;
            variance += d * d;
        }
        variance /= input.Length;

        var invStd = 1f / MathF.Sqrt(variance + epsilon);
        for (var i = 0; i < input.Length; i++)
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