using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Dotllm.Loading;
using Dotllm.Models;
using Dotllm.Sampling;
using Dotllm.Tensors;
using Dotllm.Tokenization;

namespace Dotllm.Inference;

public sealed class InferenceEngine
{
    private readonly LoadedModel _model;
    private readonly TransformerConfig _cfg;
    private readonly TensorNameResolver _tn;
    private readonly IComputeBackend _backend;
    private readonly BpeTokenizer _tokenizer;
    private readonly float[] _ropeFreqs;
    private KvCache _kvCache;
    private InferenceBuffers _buffers;
    private ConvStateCache? _convCache;

    public InferenceEngine(LoadedModel model) : this(model, new CpuBackend()) { }

    internal InferenceEngine(LoadedModel model, IComputeBackend backend)
    {
        _model = model;
        _cfg = model.Config;
        _tn = model.TensorNames;
        _backend = backend;
        _tokenizer = model.Tokenizer;
        _buffers = new InferenceBuffers(_cfg);
        _kvCache = new KvCache(_cfg.LayerCount, _cfg.KvDim, _cfg.ContextLength);
        _convCache = _cfg.HasConvLayers ? new ConvStateCache(_cfg.LayerCount, _cfg.HiddenSize, _cfg.ConvKernelSize) : null;
        _ropeFreqs = PrecomputeRopeFrequencies(_cfg);
    }

    private static float[] PrecomputeRopeFrequencies(TransformerConfig cfg)
    {
        var actualRotaryDim = cfg.RopeDimensionCount > 0 ? cfg.RopeDimensionCount : cfg.HeadDim;
        var halfDim = actualRotaryDim / 2;
        var freqs = new float[halfDim];
        for (var i = 0; i < halfDim; i++)
        {
            var freq = 1f / MathF.Pow(cfg.RopeFreqBase, 2f * i / actualRotaryDim);

            if (cfg.RopeScalingType == RopeScalingType.Linear && cfg.RopeScalingFactor > 1f)
                freq /= cfg.RopeScalingFactor;

            freqs[i] = freq;
        }
        return freqs;
    }

    public async IAsyncEnumerable<int> Generate(
        int[] promptTokens,
        GenerationOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        _kvCache.Reset();
        _convCache?.Reset();

        foreach (var token in promptTokens)
        {
            ProcessToken(token, 0, _kvCache, _buffers);
            _kvCache.Advance();
        }

        var lastToken = promptTokens.Length > 0 ? promptTokens[^1] : 0;
        var generated = 0;
        var stopSequences = options.StopSequences;
        var generatedText = new System.Text.StringBuilder();
        var recentTokens = new List<int>(promptTokens);
        var repeatWindowSize = options.Sampling.RepeatPenaltyWindowSize;

        while (generated < options.MaxTokens && !cancellationToken.IsCancellationRequested)
        {
            var logits = ProcessToken(lastToken, _kvCache.CurrentPosition, _kvCache, _buffers);

            if (_cfg.FinalLogitSoftcap.HasValue)
                _backend.Softcap(logits, _cfg.FinalLogitSoftcap.Value);

            var nextToken = SampleToken(logits, options.Sampling, recentTokens);
            _kvCache.Advance();
            recentTokens.Add(nextToken);
            if (recentTokens.Count > repeatWindowSize + 1)
                recentTokens.RemoveRange(0, recentTokens.Count - repeatWindowSize - 1);
            yield return nextToken;

            if (nextToken == _cfg.EosTokenId)
                yield break;

            if (stopSequences is { Length: > 0 })
            {
                generatedText.Append(_tokenizer.Decode(nextToken));
                var text = generatedText.ToString();
                foreach (var stop in stopSequences)
                {
                    if (text.Contains(stop, StringComparison.Ordinal))
                        yield break;
                }
            }

            lastToken = nextToken;
            generated++;
        }
    }

    private float[] ProcessToken(int token, int position, KvCache kvCache, InferenceBuffers buf)
    {
        GetEmbedding(token, buf.HiddenState);

        if (_cfg.EmbeddingScale != 1f)
            _backend.Scale(buf.HiddenState, _cfg.EmbeddingScale);

        for (var layer = 0; layer < _cfg.LayerCount; layer++)
            ProcessLayer(layer, position, buf.HiddenState, kvCache, buf);

        ApplyFinalNorm(buf.HiddenState);
        return ComputeLogits(buf.HiddenState, buf.Logits);
    }

    private void GetEmbedding(int token, float[] output)
    {
        var emb = _model.GetTensor("token_embd.weight");
        var rowElements = emb.ColumnCount;
        var rowBytes = (int)TensorSize.ByteCount(emb.ElementType, (ulong)rowElements);
        var offset = token * rowBytes;
        var src = emb.Data.Span.Slice(offset, rowBytes);
        _backend.DequantizeToFloat(src, output.AsSpan(0, rowElements), emb.ElementType, 1, rowElements);
    }

    private void ProcessLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var layerType = _cfg.HasConvLayers && layer < _cfg.LayerTypes.Length
            ? _cfg.LayerTypes[layer]
            : LayerType.Attention;

        switch (_cfg.Template)
        {
            case ExecutionTemplate.LlamaLike:
                ProcessLlamaLikeLayer(layer, position, hiddenState, kvCache, buf);
                break;
            case ExecutionTemplate.GptNeoXLike:
                ProcessGptNeoXLikeLayer(layer, position, hiddenState, kvCache, buf);
                break;
            case ExecutionTemplate.GemmaLike:
                ProcessGemmaLikeLayer(layer, position, hiddenState, kvCache, buf);
                break;
            case ExecutionTemplate.Lfm2Like:
                if (layerType == LayerType.Attention)
                    ProcessLlamaLikeLayer(layer, position, hiddenState, kvCache, buf);
                else
                    ProcessConvLayer(layer, position, hiddenState, kvCache, buf);
                break;
        }
    }

    private void ProcessLlamaLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        _backend.Add(hs, buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize));
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private void ProcessGptNeoXLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        ApplyNorm(hs, buf.NormBuf2, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf2, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize), buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private void ProcessGemmaLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        _backend.Add(hs, buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize));
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, layer, isAttnPostNorm: true);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, layer, isAttnPostNorm: false);
    }

    private void ProcessConvLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        var layerTensors = _tn.ResolveLayer(layer, _cfg);

        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);

        if (layerTensors.ConvWeight is not null && _convCache is not null)
        {
            var convWeight = _model.GetTensor(layerTensors.ConvWeight.Name);
            var convWeightData = MemoryMarshal.Cast<byte, float>(convWeight.Data.Span);
            var kernelSize = _cfg.ConvKernelSize;

            var convInput = _convCache.BuildInput(layer, buf.NormBuf, position);
            _backend.Conv1D(convInput, convWeightData, buf.ConvBuf.AsSpan(0, _cfg.HiddenSize), kernelSize, _cfg.HiddenSize);
            _convCache.Store(layer, buf.NormBuf);

            if (layerTensors.ConvBias is not null)
            {
                var biasData = MemoryMarshal.Cast<byte, float>(_model.GetTensor(layerTensors.ConvBias.Name).Data.Span);
                _backend.Add(buf.ConvBuf.AsSpan(0, _cfg.HiddenSize), biasData);
            }

            _backend.Add(hs, buf.ConvBuf.AsSpan(0, _cfg.HiddenSize));
        }
        else
        {
            _convCache?.Store(layer, buf.NormBuf);
        }

        ApplyNorm(hs, buf.NormBuf2, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf2, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private void AttentionForward(int layer, int position, ReadOnlySpan<float> input, KvCache kvCache, InferenceBuffers buf)
    {
        var headDim = _cfg.HeadDim;
        var headCount = _cfg.HeadCount;
        var headCountKv = _cfg.HeadCountKv;
        var qDim = _cfg.QDim;
        var kvDim = _cfg.KvDim;
        var hidden = _cfg.HiddenSize;

        var q = buf.QBuf.AsSpan(0, qDim);
        var k = buf.KBuf.AsSpan(0, kvDim);
        var v = buf.VBuf.AsSpan(0, kvDim);

        var layerTensors = _tn.ResolveLayer(layer, _cfg);

        if (_cfg.QkvLayout == QkvLayout.Fused && layerTensors.FusedQkvWeight is not null)
        {
            var fusedTensor = _model.GetTensor(layerTensors.FusedQkvWeight.Name);
            var fusedCols = qDim + 2 * kvDim;
            MatMulFromTensor(input, fusedTensor, q[..Math.Min(qDim, fusedCols)], fusedCols);
        }
        else
        {
            if (layerTensors.Q is not null)
            {
                var qTensor = _model.GetTensor(layerTensors.Q.Name);
                var qCols = (int)(qTensor.ElementCount / (ulong)hidden);
                MatMulFromTensor(input, qTensor, q, qCols);
            }
            if (layerTensors.K is not null)
            {
                var kTensor = _model.GetTensor(layerTensors.K.Name);
                var kCols = (int)(kTensor.ElementCount / (ulong)hidden);
                MatMulFromTensor(input, kTensor, k, kCols);
            }
            if (layerTensors.V is not null)
            {
                var vTensor = _model.GetTensor(layerTensors.V.Name);
                var vCols = (int)(vTensor.ElementCount / (ulong)hidden);
                MatMulFromTensor(input, vTensor, v, vCols);
            }
        }

        for (var h = 0; h < headCount; h++)
        {
            var qHead = q.Slice(h * headDim, headDim);
            _backend.ApplyRoPE(qHead, qHead, headDim, position, _ropeFreqs);
        }

        for (var h = 0; h < headCountKv; h++)
        {
            var kHead = k.Slice(h * headDim, headDim);
            var vHead = v.Slice(h * headDim, headDim);
            _backend.ApplyRoPE(kHead, kHead, headDim, position, _ropeFreqs);
            kHead.CopyTo(kvCache.GetKeySlot(layer, position).Slice(h * headDim, headDim));
            vHead.CopyTo(kvCache.GetValueSlot(layer, position).Slice(h * headDim, headDim));
        }

        var seqLen = position + 1;
        var attnStart = _cfg.SlidingWindow > 0 ? Math.Max(0, seqLen - _cfg.SlidingWindow) : 0;
        var result = buf.AttnResultBuf.AsSpan(0, qDim);
        var scoreBuf = buf.AttnOutBuf.AsSpan(0, hidden);
        var keys = kvCache.GetKeys(layer);
        var values = kvCache.GetValues(layer);
        var attnLen = seqLen - attnStart;
        var groupSize = headCount / headCountKv;
        var invSqrtHeadDim = 1f / MathF.Sqrt(headDim);

        for (var kvH = 0; kvH < headCountKv; kvH++)
        {
            var kvKeyOff = kvH * headDim;
            var kvValOff = kvH * headDim;

            for (var qInGroup = 0; qInGroup < groupSize; qInGroup++)
            {
                var h = kvH * groupSize + qInGroup;
                var qSlice = q.Slice(h * headDim, headDim);

                for (var s = attnStart; s < seqLen; s++)
                {
                    var dot = 0f;
                    var kOff = s * kvDim + kvKeyOff;
                    for (var d = 0; d < headDim; d++)
                        dot += qSlice[d] * keys[kOff + d];
                    scoreBuf[s - attnStart] = dot * invSqrtHeadDim;
                }

                _backend.Softmax(scoreBuf[..attnLen], _cfg.AttnLogitSoftcap);

                for (var d = 0; d < headDim; d++)
                {
                    var val = 0f;
                    for (var s = 0; s < attnLen; s++)
                        val += scoreBuf[s] * values[(attnStart + s) * kvDim + kvValOff + d];
                    result[h * headDim + d] = val;
                }
            }
        }

        var outWeight = layerTensors.AttnOutputWeight;
        var outTensor = _model.GetTensor(outWeight.Name);
        var outCols = (int)(outTensor.ElementCount / (ulong)(headCount * headDim));
        MatMulFromTensor(result, outTensor, buf.AttnResultBuf.AsSpan(0, hidden), outCols);
    }

    private void FfnForward(int layer, ReadOnlySpan<float> input, float[] ffnBuf, float[] resultBuf)
    {
        var layerTensors = _tn.ResolveLayer(layer, _cfg);
        var hidden = _cfg.HiddenSize;

        if (_cfg.FfnType == FfnType.SwiGLU || _cfg.FfnType == FfnType.GeGLU)
        {
            var gateTensor = _model.GetTensor(layerTensors.FfnGateWeight!.Name);
            var upTensor = _model.GetTensor(layerTensors.FfnUpWeight!.Name);
            var downTensor = _model.GetTensor(layerTensors.FfnDownWeight.Name);

            var gateCols = (int)(gateTensor.ElementCount / (ulong)hidden);
            var upCols = (int)(upTensor.ElementCount / (ulong)hidden);

            MatMulFromTensor(input, gateTensor, ffnBuf.AsSpan(0, gateCols), gateCols);

            if (_cfg.FfnType == FfnType.SwiGLU)
                _backend.SiluInPlace(ffnBuf.AsSpan(0, gateCols));
            else
                _backend.Gelu(ffnBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(0, gateCols));

            MatMulFromTensor(input, upTensor, resultBuf.AsSpan(0, upCols), upCols);
            _backend.Mul(ffnBuf.AsSpan(0, gateCols), resultBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(0, gateCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, gateCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
        else
        {
            var upTensor = _model.GetTensor(layerTensors.FfnUpWeight.Name);
            var downTensor = _model.GetTensor(layerTensors.FfnDownWeight.Name);
            var upCols = (int)(upTensor.ElementCount / (ulong)hidden);

            MatMulFromTensor(input, upTensor, ffnBuf.AsSpan(0, upCols), upCols);
            _backend.Gelu(ffnBuf.AsSpan(0, upCols), ffnBuf.AsSpan(0, upCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, upCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
    }

    private void ApplyNorm(Span<float> input, float[] output, int layer, bool isAttnNorm)
    {
        var tensors = _tn.ResolveLayer(layer, _cfg);
        var normWeightInfo = isAttnNorm ? tensors.NormWeight : tensors.FfnNormWeight ?? tensors.NormWeight;
        var weights = _model.GetDequantizedWeights(normWeightInfo.Name);

        var outSpan = output.AsSpan(0, _cfg.HiddenSize);

        if (_cfg.NormType == NormType.RmsNorm)
        {
            _backend.RmsNorm(input, weights, outSpan, _cfg.NormEpsilon);
        }
        else
        {
            var biasInfo = isAttnNorm ? _tn.TryLayerNormBias(layer) : null;
            var bias = biasInfo is not null ? _model.GetDequantizedWeights(biasInfo.Name) : new float[_cfg.HiddenSize];
            _backend.LayerNorm(input, weights, bias, outSpan, _cfg.NormEpsilon);
        }
    }

    private void ApplyPostNorm(Span<float> hiddenState, int layer, bool isAttnPostNorm)
    {
        var tensors = _tn.ResolveLayer(layer, _cfg);
        var postNormInfo = isAttnPostNorm ? tensors.PostAttentionNormWeight : tensors.PostFfnNormWeight;
        if (postNormInfo is null) return;

        var weights = _model.GetDequantizedWeights(postNormInfo.Name);
        var tmp = hiddenState.ToArray();
        _backend.RmsNorm(tmp, weights, hiddenState, _cfg.NormEpsilon);
    }

    private void ApplyFinalNorm(Span<float> hiddenState)
    {
        var weights = _model.GetDequantizedWeights("output_norm.weight");
        var tmp = hiddenState.ToArray();
        _backend.RmsNorm(tmp, weights, hiddenState, _cfg.NormEpsilon);
    }

    private float[] ComputeLogits(ReadOnlySpan<float> hiddenState, float[] logits)
    {
        Tensor outputTensor;
        if (_model.TryGetTensor("output.weight", out var outTensor) && outTensor is not null)
            outputTensor = outTensor;
        else
            outputTensor = _model.GetTensor("token_embd.weight");

        var vocabSize = _cfg.VocabSize;
        var rowBytes = (int)TensorSize.ByteCount(outputTensor.ElementType, (ulong)_cfg.HiddenSize);

        float[]? rented = null;
        Span<float> tmp = _cfg.HiddenSize <= 4096 ? stackalloc float[_cfg.HiddenSize] : (rented = System.Buffers.ArrayPool<float>.Shared.Rent(_cfg.HiddenSize));

        try
        {
            for (var i = 0; i < Math.Min(vocabSize, logits.Length); i++)
            {
                var rowSrc = outputTensor.Data.Span.Slice(i * rowBytes, rowBytes);
                _backend.DequantizeToFloat(rowSrc, tmp, outputTensor.ElementType, 1, _cfg.HiddenSize);
                var dot = 0f;
                for (var j = 0; j < _cfg.HiddenSize; j++)
                    dot += hiddenState[j] * tmp[j];
                logits[i] = dot;
            }
        }
        finally
        {
            if (rented is not null)
                System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }

        return logits;
    }

    private void MatMulFromTensor(ReadOnlySpan<float> input, Tensor weight, Span<float> output, int cols)
    {
        if (weight.ElementType == GgmlType.F32)
        {
            var weightData = MemoryMarshal.Cast<byte, float>(weight.Data.Span);
            _backend.MatMulF32(input, weightData, output, _cfg.HiddenSize, cols, weight.Name);
        }
        else
        {
            _backend.MatMul(input, weight.Data.Span, output, weight.ElementType, _cfg.HiddenSize, cols, weight.Name);
        }
    }

    private int SampleToken(ReadOnlySpan<float> logits, SamplingOptions options, List<int> recentTokens)
    {
        if (options.Temperature <= 0f)
            return _backend.ArgMax(logits);

        var length = logits.Length;
        var scaled = length <= 4096 ? stackalloc float[length] : new float[length];
        logits.CopyTo(scaled);

        if (options.RepeatPenalty != 1f && recentTokens.Count > 0 && options.RepeatPenaltyWindowSize > 0)
        {
            var windowStart = Math.Max(0, recentTokens.Count - options.RepeatPenaltyWindowSize);
            for (var i = windowStart; i < recentTokens.Count; i++)
            {
                var tokenId = recentTokens[i];
                if (tokenId >= 0 && tokenId < length)
                {
                    if (scaled[tokenId] > 0)
                        scaled[tokenId] /= options.RepeatPenalty;
                    else
                        scaled[tokenId] *= options.RepeatPenalty;
                }
            }
        }

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
            var topPIndices = new int[length];
            var topPValues = new float[length];
            var nucleusSize = FindNucleus(scaled, options.TopP, topPIndices, topPValues);
            if (nucleusSize < length)
            {
                var nucleusScaled = (Span<float>)topPValues.AsSpan(0, nucleusSize);
                _backend.Scale(nucleusScaled, 1f / nucleusScaled[0..nucleusSize].ToArray().Sum());
                return topPIndices[SampleFromProbabilities(nucleusScaled, options.Seed)];
            }
        }

        return SampleFromProbabilities(scaled, options.Seed);
    }

    private static void FindTopK(Span<float> logits, int k, out int[] indices, out Span<float> values, int length)
    {
        indices = new int[k];
        var tempValues = new float[k];
        var tempIndices = new int[length];
        for (var i = 0; i < length; i++) tempIndices[i] = i;

        for (var i = 0; i < k; i++)
        {
            var maxIdx = i;
            for (var j = i + 1; j < length; j++)
            {
                if (logits[tempIndices[j]] > logits[tempIndices[maxIdx]])
                    maxIdx = j;
            }
            (tempIndices[i], tempIndices[maxIdx]) = (tempIndices[maxIdx], tempIndices[i]);
            tempValues[i] = logits[tempIndices[i]];
        }

        for (var i = 0; i < k; i++) indices[i] = tempIndices[i];
        values = tempValues.AsSpan(0, k);
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