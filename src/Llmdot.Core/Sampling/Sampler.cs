using Llmdot.Inference;
using Llmdot.Tensors.Numeric;

namespace Llmdot.Sampling;

/// <summary>
/// Implements token sampling strategies for language model inference,
/// including temperature scaling, top-k filtering, top-p (nucleus) sampling,
/// and repeat penalty.
/// </summary>
public sealed class Sampler
{
    private readonly IComputeBackend _backend;

    public Sampler(IComputeBackend backend)
    {
        _backend = backend;
    }

    /// <summary>
    /// Samples a single token from the logits using the given sampling options
    /// and recent token history (for repeat penalty).
    /// </summary>
    public int Sample(ReadOnlySpan<float> logits, SamplingOptions options, float[] samplingBuf, int[] samplingIdxBuf, List<int> recentTokens)
    {
        if (options.Temperature <= 0f)
            return _backend.ArgMax(logits);

        var length = logits.Length;
        var scaled = samplingBuf.AsSpan(0, length);
        logits.CopyTo(scaled);

        ApplyRepeatPenalty(scaled, options, recentTokens);
        _backend.Scale(scaled, 1f / options.Temperature);

        if (options.TopK > 0 && options.TopK < length)
        {
            var k = Math.Min(options.TopK, length);
            FindTopK(scaled, k, out var topIndices, out var topValues, length);
            _backend.Softmax(topValues);
            return topIndices[SampleFromProbabilities(topValues, options.Seed)];
        }

        _backend.Softmax(scaled);

        if (options.TopP < 1f)
        {
            var topPIndices = samplingIdxBuf;
            var topPValues = samplingBuf;
            var nucleusSize = FindNucleus(scaled, options.TopP, topPIndices, topPValues);
            if (nucleusSize < length)
            {
                var nucleusScaled = topPValues.AsSpan(0, nucleusSize);
                var sum = 0f;
                for (var i = 0; i < nucleusSize; i++) sum += nucleusScaled[i];
                _backend.Scale(nucleusScaled, 1f / sum);
                return topPIndices[SampleFromProbabilities(nucleusScaled, options.Seed)];
            }
        }

        return SampleFromProbabilities(scaled, options.Seed);
    }

    private static void ApplyRepeatPenalty(Span<float> logits, SamplingOptions options, List<int> recentTokens)
    {
        if (options.RepeatPenalty == 1f || recentTokens.Count == 0 || options.RepeatPenaltyWindowSize <= 0)
            return;

        var windowStart = Math.Max(0, recentTokens.Count - options.RepeatPenaltyWindowSize);
        for (var i = windowStart; i < recentTokens.Count; i++)
        {
            var tokenId = recentTokens[i];
            if (tokenId >= 0 && tokenId < logits.Length)
            {
                if (logits[tokenId] > 0)
                    logits[tokenId] /= options.RepeatPenalty;
                else
                    logits[tokenId] *= options.RepeatPenalty;
            }
        }
    }

    private static void FindTopK(Span<float> logits, int k, out int[] indices, out Span<float> values, int length)
    {
        var topIdx = new int[k];
        var topVal = new float[k];

        for (var i = 0; i < k; i++)
        {
            topIdx[i] = i;
            topVal[i] = logits[i];
        }

        for (var i = k; i < length; i++)
        {
            var minIdx = 0;
            for (var j = 1; j < k; j++)
            {
                if (topVal[j] < topVal[minIdx])
                    minIdx = j;
            }

            if (logits[i] > topVal[minIdx])
            {
                topIdx[minIdx] = i;
                topVal[minIdx] = logits[i];
            }
        }

        for (var i = 1; i < k; i++)
        {
            var keyIdx = topIdx[i];
            var keyVal = topVal[i];
            var j = i - 1;
            while (j >= 0 && topVal[j] < keyVal)
            {
                topIdx[j + 1] = topIdx[j];
                topVal[j + 1] = topVal[j];
                j--;
            }
            topIdx[j + 1] = keyIdx;
            topVal[j + 1] = keyVal;
        }

        indices = topIdx;
        values = topVal.AsSpan(0, k);
    }

    private static int FindNucleus(ReadOnlySpan<float> probabilities, float topP, int[] indices, float[] values)
    {
        var length = probabilities.Length;
        for (var i = 0; i < length; i++) indices[i] = i;

        for (var i = 0; i < length - 1; i++)
        {
            var maxIdx = i;
            for (var j = i + 1; j < length; j++)
            {
                if (probabilities[indices[j]] > probabilities[indices[maxIdx]])
                    maxIdx = j;
            }
            if (maxIdx != i)
                (indices[i], indices[maxIdx]) = (indices[maxIdx], indices[i]);
        }

        var cumulative = 0f;
        var count = 0;
        for (var i = 0; i < length; i++)
        {
            values[i] = probabilities[indices[i]];
            cumulative += values[i];
            count++;
            if (cumulative >= topP) break;
        }
        return count;
    }

    private static int SampleFromProbabilities(ReadOnlySpan<float> probabilities, int seed)
    {
        var rng = seed >= 0 ? new Random(seed) : Random.Shared;
        var r = rng.NextSingle();
        var cumulative = 0f;
        for (var i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (r <= cumulative)
                return i;
        }
        return probabilities.Length - 1;
    }
}