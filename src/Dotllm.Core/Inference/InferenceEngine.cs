using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Dotllm.Loading;
using Dotllm.Models;
using Dotllm.Sampling;
using Dotllm.Tensors;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Inference;

public sealed class InferenceEngine
{
    private readonly LoadedModel _model;
    private readonly TransformerConfig _cfg;
    private readonly TensorNameResolver _tn;
    private readonly IComputeBackend _backend;

    public InferenceEngine(LoadedModel model) : this(model, new CpuBackend()) { }

    internal InferenceEngine(LoadedModel model, IComputeBackend backend)
    {
        _model = model;
        _cfg = model.Config;
        _tn = model.TensorNames;
        _backend = backend;
    }

    public async IAsyncEnumerable<int> Generate(
        int[] promptTokens,
        GenerationOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var kvCache = new KvCache(_cfg.LayerCount, _cfg.KvDim, _cfg.ContextLength);
        var buffers = new InferenceBuffers(_cfg);

        foreach (var token in promptTokens)
        {
            ProcessToken(token, 0, kvCache, buffers);
            kvCache.Advance();
        }

        var lastToken = promptTokens.Length > 0 ? promptTokens[^1] : 0;
        var generated = 0;

        while (generated < options.MaxTokens && !cancellationToken.IsCancellationRequested)
        {
            var logits = ProcessToken(lastToken, kvCache.CurrentPosition, kvCache, buffers);

            if (_cfg.FinalLogitSoftcap.HasValue)
                VectorMath.Softcap(logits, _cfg.FinalLogitSoftcap.Value);

            var nextToken = SampleToken(logits, options.Sampling);
            kvCache.Advance();
            yield return nextToken;

            if (nextToken == _cfg.EosTokenId)
                yield break;

            lastToken = nextToken;
            generated++;
        }
    }

    private float[] ProcessToken(int token, int position, KvCache kvCache, InferenceBuffers buf)
    {
        GetEmbedding(token, buf.HiddenState);

        if (_cfg.EmbeddingScale != 1f)
            VectorMath.Scale(buf.HiddenState, _cfg.EmbeddingScale);

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
        TensorOps.DequantizeToFloat(src, output.AsSpan(0, rowElements), emb.ElementType, 1, rowElements);
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
                    ProcessConvLayer(layer, hiddenState, buf);
                break;
        }
    }

    private void ProcessLlamaLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        VectorMath.Add(hs, buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize));
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        VectorMath.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private void ProcessGptNeoXLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        ApplyNorm(hs, buf.NormBuf2, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf2, buf.FfnBuf, buf.FfnResultBuf);
        VectorMath.Add(buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize), buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
        VectorMath.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private void ProcessGemmaLikeLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);
        AttentionForward(layer, position, buf.NormBuf, kvCache, buf);
        VectorMath.Add(hs, buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize));
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, layer, isAttnPostNorm: true);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        VectorMath.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, layer, isAttnPostNorm: false);
    }

    private void ProcessConvLayer(int layer, float[] hiddenState, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        var layerTensors = _tn.ResolveLayer(layer, _cfg);
        if (layerTensors.FfnGateWeight is null) return;

        var gateTensor = _model.GetTensor(layerTensors.FfnGateWeight.Name);
        var upTensor = _model.GetTensor(layerTensors.FfnUpWeight!.Name);
        var downTensor = _model.GetTensor(layerTensors.FfnDownWeight.Name);

        var gateCols = (int)(gateTensor.ElementCount / (ulong)_cfg.HiddenSize);
        var upCols = (int)(upTensor.ElementCount / (ulong)_cfg.HiddenSize);

        MatMulFromTensor(hs, gateTensor, buf.FfnBuf.AsSpan(0, gateCols), gateCols);
        VectorMath.SiluInPlace(buf.FfnBuf.AsSpan(0, gateCols));
        MatMulFromTensor(hs, upTensor, buf.ConvBuf.AsSpan(0, upCols), upCols);
        VectorMath.Mul(buf.FfnBuf.AsSpan(0, gateCols), buf.ConvBuf.AsSpan(0, gateCols), buf.FfnBuf.AsSpan(0, gateCols));
        MatMulFromTensor(buf.FfnBuf.AsSpan(0, gateCols), downTensor, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize), _cfg.HiddenSize);
        VectorMath.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
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
            TensorOps.ApplyRoPE(qHead, qHead, headDim, position, _cfg.RopeFreqBase, _cfg.RopeDimensionCount);
        }

        for (var h = 0; h < headCountKv; h++)
        {
            var kHead = k.Slice(h * headDim, headDim);
            var vHead = v.Slice(h * headDim, headDim);
            TensorOps.ApplyRoPE(kHead, kHead, headDim, position, _cfg.RopeFreqBase, _cfg.RopeDimensionCount);
            kHead.CopyTo(kvCache.GetKeySlot(layer, position).Slice(h * headDim, headDim));
            vHead.CopyTo(kvCache.GetValueSlot(layer, position).Slice(h * headDim, headDim));
        }

        var seqLen = position + 1;
        var result = buf.AttnResultBuf.AsSpan(0, qDim);
        var scoreBuf = buf.AttnOutBuf.AsSpan(0, hidden);

        for (var h = 0; h < headCount; h++)
        {
            var kvGroup = h / (headCount / headCountKv);
            var qSlice = q.Slice(h * headDim, headDim);
            var keys = kvCache.GetKeys(layer);
            var values = kvCache.GetValues(layer);

            for (var s = 0; s < seqLen; s++)
            {
                var dot = 0f;
                for (var d = 0; d < headDim; d++)
                    dot += qSlice[d] * keys[s * kvDim + kvGroup * headDim + d];
                scoreBuf[s] = dot / MathF.Sqrt(headDim);
            }

            VectorMath.Softmax(scoreBuf[..seqLen], _cfg.AttnLogitSoftcap);

            for (var d = 0; d < headDim; d++)
            {
                var val = 0f;
                for (var s = 0; s < seqLen; s++)
                    val += scoreBuf[s] * values[s * kvDim + kvGroup * headDim + d];
                result[h * headDim + d] = val;
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
                VectorMath.SiluInPlace(ffnBuf.AsSpan(0, gateCols));
            else
                VectorMath.Gelu(ffnBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(0, gateCols));

            MatMulFromTensor(input, upTensor, resultBuf.AsSpan(0, upCols), upCols);
            VectorMath.Mul(ffnBuf.AsSpan(0, gateCols), resultBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(0, gateCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, gateCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
        else
        {
            var upTensor = _model.GetTensor(layerTensors.FfnUpWeight.Name);
            var downTensor = _model.GetTensor(layerTensors.FfnDownWeight.Name);
            var upCols = (int)(upTensor.ElementCount / (ulong)hidden);

            MatMulFromTensor(input, upTensor, ffnBuf.AsSpan(0, upCols), upCols);
            VectorMath.Gelu(ffnBuf.AsSpan(0, upCols), ffnBuf.AsSpan(0, upCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, upCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
    }

    private void ApplyNorm(Span<float> input, float[] output, int layer, bool isAttnNorm)
    {
        var tensors = _tn.ResolveLayer(layer, _cfg);
        var normWeightInfo = isAttnNorm ? tensors.NormWeight : tensors.FfnNormWeight ?? tensors.NormWeight;
        var normWeight = _model.GetTensor(normWeightInfo.Name);
        var weights = DequantizeWeights(normWeight);

        var outSpan = output.AsSpan(0, _cfg.HiddenSize);

        if (_cfg.NormType == NormType.RmsNorm)
        {
            VectorMath.RmsNorm(input, weights, outSpan, _cfg.NormEpsilon);
        }
        else
        {
            var biasInfo = isAttnNorm ? _tn.TryLayerNormBias(layer) : null;
            var bias = biasInfo is not null ? DequantizeWeights(_model.GetTensor(biasInfo.Name)) : new float[_cfg.HiddenSize];
            VectorMath.LayerNorm(input, weights, bias, outSpan, _cfg.NormEpsilon);
        }
    }

    private void ApplyPostNorm(Span<float> hiddenState, int layer, bool isAttnPostNorm)
    {
        var tensors = _tn.ResolveLayer(layer, _cfg);
        var postNormInfo = isAttnPostNorm ? tensors.PostAttentionNormWeight : tensors.PostFfnNormWeight;
        if (postNormInfo is null) return;

        var normWeight = _model.GetTensor(postNormInfo.Name);
        var weights = DequantizeWeights(normWeight);
        var tmp = hiddenState.ToArray();
        VectorMath.RmsNorm(tmp, weights, hiddenState, _cfg.NormEpsilon);
    }

    private void ApplyFinalNorm(Span<float> hiddenState)
    {
        var normTensor = _model.GetTensor("output_norm.weight");
        var weights = DequantizeWeights(normTensor);
        var tmp = hiddenState.ToArray();
        VectorMath.RmsNorm(tmp, weights, hiddenState, _cfg.NormEpsilon);
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
                TensorOps.DequantizeToFloat(rowSrc, tmp, outputTensor.ElementType, 1, _cfg.HiddenSize);
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
            TensorOps.MatMulF32(input, weightData, output, _cfg.HiddenSize, cols);
        }
        else
        {
            TensorOps.MatMul(input, weight.Data.Span, output, weight.ElementType, _cfg.HiddenSize, cols);
        }
    }

    private static float[] DequantizeWeights(Tensor tensor)
    {
        var size = tensor.RowCount > 0 ? tensor.RowCount * tensor.ColumnCount : (int)tensor.ElementCount;
        var result = new float[size];
        var byteCount = (int)TensorSize.ByteCount(tensor.ElementType, (ulong)size);
        TensorOps.DequantizeToFloat(tensor.Data.Span[..byteCount], result, tensor.ElementType, tensor.RowCount > 0 ? tensor.RowCount : 1, size / Math.Max(tensor.RowCount, 1));
        return result;
    }

    private static int SampleToken(ReadOnlySpan<float> logits, SamplingOptions options)
    {
        if (options.Temperature <= 0f)
            return VectorMath.ArgMax(logits);

        var scaled = logits.ToArray();
        VectorMath.Scale(scaled, 1f / options.Temperature);
        VectorMath.Softmax(scaled);

        var rng = options.Seed >= 0 ? new Random(options.Seed) : Random.Shared;
        var r = rng.NextSingle();
        var cumulative = 0f;
        for (var i = 0; i < scaled.Length; i++)
        {
            cumulative += scaled[i];
            if (r <= cumulative)
                return i;
        }
        return scaled.Length - 1;
    }
}