using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Llmdot.Loading;
using Llmdot.Models;
using Llmdot.Sampling;
using Llmdot.Tensors;
using Llmdot.Tensors.Numeric;
using Llmdot.Tokenization;

// ReSharper disable InconsistentNaming

namespace Llmdot.Inference;

/// <summary>
/// Runs autoregressive inference on a loaded GGUF model, producing token sequences
/// from prompt inputs. Uses the provided <see cref="IComputeBackend"/> for tensor operations.
/// </summary>
public sealed class InferenceEngine
{
    private readonly LoadedModel _model;
    private readonly TransformerConfig _cfg;
    private readonly TensorNameResolver _tn;
    private readonly IComputeBackend _backend;
    private readonly BpeTokenizer _tokenizer;
    private readonly float[] _ropeFreqs;
    private readonly Sampler _sampler;
    private readonly MoeRouter? _moeRouter;
    private KvCache _kvCache;
    private InferenceBuffers _buffers;
    private ConvStateCache? _convCache;
    private readonly Dictionary<string, float[]> _convWeightCache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Creates an inference engine using the CPU backend.
    /// </summary>
    public InferenceEngine(LoadedModel model) : this(model, new CpuBackend()) { }

    /// <summary>
    /// Creates an inference engine with a custom compute backend (e.g., GPU).
    /// </summary>
    public InferenceEngine(LoadedModel model, IComputeBackend backend)
    {
        _model = model;
        _cfg = model.Config;
        _tn = model.TensorNames;
        _backend = backend;
        _tokenizer = model.Tokenizer;
        _sampler = new Sampler(backend);
        _buffers = new InferenceBuffers(_cfg);
        _kvCache = new KvCache(_cfg.LayerCount, _cfg.KvDim, _cfg.ContextLength);
        _convCache = _cfg.HasConvLayers ? new ConvStateCache(_cfg.LayerCount, _cfg.HiddenSize, _cfg.ConvKernelSize) : null;
        _ropeFreqs = PrecomputeRopeFrequencies(_cfg);
        _moeRouter = _cfg.ExpertCount > 0 ? new MoeRouter(backend) : null;
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

        float[]? logits = null;

        for (var i = 0; i < promptTokens.Length; i++)
        {
            logits = ProcessToken(promptTokens[i], i, _kvCache, _buffers);
            _kvCache.Advance();
        }

        if (logits is null)
        {
            var initialToken = _cfg.BosTokenId > 0 ? _cfg.BosTokenId : 0;
            logits = ProcessToken(initialToken, 0, _kvCache, _buffers);
            _kvCache.Advance();
        }

        var generated = 0;
        var stopSequences = options.StopSequences;
        var generatedText = new System.Text.StringBuilder();
        var recentTokens = new List<int>(promptTokens);
        var repeatWindowSize = options.Sampling.RepeatPenaltyWindowSize;

        while (generated < options.MaxTokens && !cancellationToken.IsCancellationRequested)
        {
            if (_cfg.FinalLogitSoftcap.HasValue)
                _backend.Softcap(logits, _cfg.FinalLogitSoftcap.Value);

            var nextToken = _sampler.Sample(logits, options.Sampling, _buffers.SamplingBuf, _buffers.SamplingIdxBuf, recentTokens);
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

            logits = ProcessToken(nextToken, _kvCache.CurrentPosition, _kvCache, _buffers);
            _kvCache.Advance();
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

        ApplyFinalNorm(buf.HiddenState, buf.NormTempBuf);
        return ComputeLogits(buf.HiddenState, buf.Logits);
    }

    private void GetEmbedding(int token, float[] output)
    {
        var emb = _model.GetTensor("token_embd.weight");
        var hiddenSize = (int)emb.Dimensions[0];
        var rowBytes = (int)TensorSize.ByteCount(emb.ElementType, (ulong)hiddenSize);
        var offset = token * rowBytes;
        var src = emb.Data.Span.Slice(offset, rowBytes);
        _backend.DequantizeToFloat(src, output.AsSpan(0, hiddenSize), emb.ElementType, 1, hiddenSize);
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
        if (_cfg.ExpertCount > 0)
            MoeFfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        else
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
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, buf.NormTempBuf, layer, isAttnPostNorm: true);
        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: false);
        FfnForward(layer, buf.NormBuf, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
        if (_cfg.HasPostNorm) ApplyPostNorm(hs, buf.NormTempBuf, layer, isAttnPostNorm: false);
    }

    private void ProcessConvLayer(int layer, int position, float[] hiddenState, KvCache kvCache, InferenceBuffers buf)
    {
        var hs = hiddenState.AsSpan(0, _cfg.HiddenSize);
        var layerTensors = _tn.ResolveLayer(layer, _cfg);

        ApplyNorm(hs, buf.NormBuf, layer, isAttnNorm: true);

        if (layerTensors.ConvInProj is not null && layerTensors.ConvWeight is not null && _convCache is not null)
        {
            var inProjTensor = _model.GetTensor(layerTensors.ConvInProj.Name);
            var convWeightTensor = _model.GetTensor(layerTensors.ConvWeight.Name);
            var convWeightFloat = GetDequantizedConvWeights(convWeightTensor);

            var hidden = _cfg.HiddenSize;
            var inProjCols = (int)(inProjTensor.ElementCount / (ulong)hidden);
            var inProjBuf = inProjCols <= buf.FfnBuf.Length ? buf.FfnBuf : new float[inProjCols];

            MatMulFromTensor(buf.NormBuf, inProjTensor, inProjBuf.AsSpan(0, inProjCols), inProjCols);

            var bChunk = inProjBuf.AsSpan(0, hidden);
            var cChunk = inProjBuf.AsSpan(hidden, hidden);
            var xChunk = inProjBuf.AsSpan(2 * hidden, hidden);

            _backend.Mul(bChunk, xChunk, buf.ConvBuf.AsSpan(0, hidden));

            var convInput = _convCache.BuildInput(layer, buf.ConvBuf.AsSpan(0, hidden), position);
            _backend.Conv1D(convInput, convWeightFloat, buf.QBuf.AsSpan(0, hidden), _cfg.ConvKernelSize, hidden);
            _convCache.Store(layer, buf.ConvBuf.AsSpan(0, hidden));

            _backend.Mul(buf.QBuf.AsSpan(0, hidden), cChunk, buf.QBuf.AsSpan(0, hidden));

            if (layerTensors.ConvOutProj is not null)
            {
                var outProjTensor = _model.GetTensor(layerTensors.ConvOutProj.Name);
                var outProjCols = (int)(outProjTensor.ElementCount / (ulong)hidden);
                MatMulFromTensor(buf.QBuf.AsSpan(0, hidden), outProjTensor, buf.AttnResultBuf.AsSpan(0, hidden), outProjCols);
            }
            else
            {
                buf.QBuf.AsSpan(0, hidden).CopyTo(buf.AttnResultBuf);
            }

            _backend.Add(hs, buf.AttnResultBuf.AsSpan(0, _cfg.HiddenSize));
        }
        else
        {
            _convCache?.Store(layer, buf.NormBuf);
        }

        ApplyNorm(hs, buf.NormBuf2, layer, isAttnNorm: false);
        if (_cfg.ExpertCount > 0)
            MoeFfnForward(layer, buf.NormBuf2, buf.FfnBuf, buf.FfnResultBuf);
        else
            FfnForward(layer, buf.NormBuf2, buf.FfnBuf, buf.FfnResultBuf);
        _backend.Add(hs, buf.FfnResultBuf.AsSpan(0, _cfg.HiddenSize));
    }

    private float[] GetDequantizedConvWeights(Tensor convWeightTensor)
    {
        var key = convWeightTensor.Name;
        if (_convWeightCache.TryGetValue(key, out var cached))
            return cached;

        var size = (int)convWeightTensor.ElementCount;
        var result = new float[size];
        if (convWeightTensor.ElementType == GgmlType.F32)
        {
            var floatData = MemoryMarshal.Cast<byte, float>(convWeightTensor.Data.Span);
            floatData.CopyTo(result);
        }
        else
        {
            var byteCount = (int)TensorSize.ByteCount(convWeightTensor.ElementType, (ulong)size);
            TensorOps.DequantizeToFloat(convWeightTensor.Data.Span[..byteCount], result, convWeightTensor.ElementType,
                convWeightTensor.RowCount,
                convWeightTensor.ColumnCount);
        }

        _convWeightCache[key] = result;
        return result;
    }

    private void AttentionForward(int layer, int position, ReadOnlySpan<float> input, KvCache kvCache, InferenceBuffers buf)
    {
        var headDim = _cfg.HeadDim;
        var headCount = _cfg.HeadCount;
        var layerHeadCountKv = _cfg.HeadCountKvPerLayer.Count > layer
            ? Math.Max(_cfg.HeadCountKvPerLayer[layer], 1)
            : _cfg.HeadCountKv;
        var qDim = _cfg.QDim;
        var layerKvDim = layerHeadCountKv * headDim;
        var hidden = _cfg.HiddenSize;

        var q = buf.QBuf.AsSpan(0, qDim);
        var k = buf.KBuf.AsSpan(0, layerKvDim);
        var v = buf.VBuf.AsSpan(0, layerKvDim);

        var layerTensors = _tn.ResolveLayer(layer, _cfg);

        if (_cfg.QkvLayout == QkvLayout.Fused && layerTensors.FusedQkvWeight is not null)
        {
            var fusedTensor = _model.GetTensor(layerTensors.FusedQkvWeight.Name);
            var fusedCols = qDim + 2 * layerKvDim;
            var fusedBuf = fusedCols <= buf.FfnBuf.Length
                ? buf.FfnBuf.AsSpan(0, fusedCols)
                : new float[fusedCols];
            MatMulFromTensor(input, fusedTensor, fusedBuf, fusedCols);
            fusedBuf[..qDim].CopyTo(q);
            fusedBuf[qDim..(qDim + layerKvDim)].CopyTo(k);
            fusedBuf[(qDim + layerKvDim)..].CopyTo(v);
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

        if (layerTensors.AttentionQNorm is not null)
        {
            var qNormWeights = _model.GetDequantizedWeights(layerTensors.AttentionQNorm.Name);
            for (var h = 0; h < headCount; h++)
                _backend.RmsNorm(q.Slice(h * headDim, headDim), qNormWeights, q.Slice(h * headDim, headDim), _cfg.NormEpsilon);
        }

        for (var h = 0; h < headCount; h++)
        {
            var qHead = q.Slice(h * headDim, headDim);
            _backend.ApplyRoPE(qHead, qHead, headDim, position, _ropeFreqs);
        }

        if (layerTensors.AttentionKNorm is not null)
        {
            var kNormWeights = _model.GetDequantizedWeights(layerTensors.AttentionKNorm.Name);
            for (var h = 0; h < layerHeadCountKv; h++)
                _backend.RmsNorm(k.Slice(h * headDim, headDim), kNormWeights, k.Slice(h * headDim, headDim), _cfg.NormEpsilon);
        }

        for (var h = 0; h < layerHeadCountKv; h++)
        {
            var kHead = k.Slice(h * headDim, headDim);
            var vHead = v.Slice(h * headDim, headDim);
            _backend.ApplyRoPE(kHead, kHead, headDim, position, _ropeFreqs);
            kHead.CopyTo(kvCache.GetKeySlot(layer, position).Slice(h * headDim, headDim));
            vHead.CopyTo(kvCache.GetValueSlot(layer, position).Slice(h * headDim, headDim));
        }

        var seqLen = position + 1;
        var attnStart = _cfg.SlidingWindow > 0 ? Math.Max(0, seqLen - _cfg.SlidingWindow) : 0;
        var attnLen = seqLen - attnStart;
        var result = buf.AttnResultBuf.AsSpan(0, qDim);
        var scoreBuf = buf.ScoreBuf.AsSpan(0, attnLen);
        var keys = kvCache.GetKeys(layer, seqLen);
        var values = kvCache.GetValues(layer, seqLen);
        var groupSize = headCount / layerHeadCountKv;
        var invSqrtHeadDim = 1f / MathF.Sqrt(headDim);
        var globalKvDim = _cfg.KvDim;
        var vecSize = Vector<float>.Count;

        for (var kvH = 0; kvH < layerHeadCountKv; kvH++)
        {
            var kvKeyOff = kvH * headDim;
            var kvValOff = kvH * headDim;

            for (var qInGroup = 0; qInGroup < groupSize; qInGroup++)
            {
                var h = kvH * groupSize + qInGroup;
                var qSlice = q.Slice(h * headDim, headDim);

                for (var s = attnStart; s < seqLen; s++)
                {
                    var kOff = s * globalKvDim + kvKeyOff;
                    var dotVec = Vector<float>.Zero;
                    var d = 0;
                    for (; d <= headDim - vecSize; d += vecSize)
                    {
                        var qVec = new Vector<float>(qSlice.Slice(d, vecSize));
                        var kVec = new Vector<float>(keys.Slice(kOff + d, vecSize));
                        dotVec += qVec * kVec;
                    }
                    var dot = Vector.Dot(dotVec, Vector<float>.One);
                    for (; d < headDim; d++)
                        dot += qSlice[d] * keys[kOff + d];
                    scoreBuf[s - attnStart] = dot * invSqrtHeadDim;
                }

                _backend.Softmax(scoreBuf[..attnLen], _cfg.AttnLogitSoftcap);

                result.Slice(h * headDim, headDim).Clear();
                var resultSlice = result.Slice(h * headDim, headDim);

                for (var s = 0; s < attnLen; s++)
                {
                    var score = scoreBuf[s];
                    var vOff = (attnStart + s) * globalKvDim + kvValOff;
                    for (var d = 0; d <= headDim - vecSize; d += vecSize)
                    {
                        var vVec = new Vector<float>(values.Slice(vOff + d, vecSize));
                        var rVec = new Vector<float>(resultSlice.Slice(d, vecSize));
                        (rVec + vVec * score).CopyTo(resultSlice.Slice(d, vecSize));
                    }
                    for (var d = headDim - headDim % vecSize; d < headDim; d++)
                        resultSlice[d] += score * values[vOff + d];
                }
            }
        }

        var outWeight = layerTensors.AttnOutputWeight!;
        var outTensor = _model.GetTensor(outWeight.Name);
        var outCols = (int)(outTensor.ElementCount / (ulong)(headCount * headDim));
        var projected = buf.AttnOutBuf.AsSpan(0, hidden);
        MatMulFromTensor(result, outTensor, projected, outCols);
        projected.CopyTo(buf.AttnResultBuf.AsSpan(0, hidden));
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

            MatMulFromTensor(input, upTensor, ffnBuf.AsSpan(gateCols, upCols), upCols);
            _backend.Mul(ffnBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(gateCols, upCols), ffnBuf.AsSpan(0, gateCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, gateCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
        else
        {
            var upTensor = _model.GetTensor(layerTensors.FfnUpWeight!.Name);
            var downTensor = _model.GetTensor(layerTensors.FfnDownWeight.Name);
            var upCols = (int)(upTensor.ElementCount / (ulong)hidden);

            MatMulFromTensor(input, upTensor, ffnBuf.AsSpan(0, upCols), upCols);
            _backend.Gelu(ffnBuf.AsSpan(0, upCols), ffnBuf.AsSpan(0, upCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, upCols), downTensor, resultBuf.AsSpan(0, hidden), hidden);
        }
    }

    private void MoeFfnForward(int layer, ReadOnlySpan<float> input, float[] ffnBuf, float[] resultBuf)
    {
        var hidden = _cfg.HiddenSize;
        var topK = _cfg.ExpertUsedCount;
        var expertCount = _cfg.ExpertCount;

        // Get MoE gate tensor
        var layerTensors = _tn.ResolveLayer(layer, _cfg);
        var gateTensor = _model.GetTensor(layerTensors.MoeGateWeight!.Name);

        // Route
        _moeRouter!.Route(
            input, gateTensor.Data.Span, gateTensor.ElementType,
            hidden, expertCount, topK,
            _buffers.MoeGateLogits, _buffers.MoeSelectedExperts, _buffers.MoeRoutingWeights);

        // Zero accumulator
        _buffers.MoeAccumulatorBuf.AsSpan(0, hidden).Clear();

        // Process each selected expert
        for (var i = 0; i < topK; i++)
        {
            var expert = _buffers.MoeSelectedExperts[i];
            var weight = _buffers.MoeRoutingWeights[i];

            var expertGate = _tn.TryExpertFfnGateWeight(layer, expert);
            var expertUp = _tn.TryExpertFfnUpWeight(layer, expert);
            var expertDown = _tn.TryExpertFfnDownWeight(layer, expert);

            if (expertGate is null || expertUp is null || expertDown is null)
                continue;

            var gateTensorE = _model.GetTensor(expertGate.Name);
            var upTensorE = _model.GetTensor(expertUp.Name);
            var downTensorE = _model.GetTensor(expertDown.Name);

            var gateCols = (int)(gateTensorE.ElementCount / (ulong)hidden);
            var upCols = (int)(upTensorE.ElementCount / (ulong)hidden);

            // SwiGLU FFN through expert tensors
            MatMulFromTensor(input, gateTensorE, ffnBuf.AsSpan(0, gateCols), gateCols);
            _backend.SiluInPlace(ffnBuf.AsSpan(0, gateCols));
            MatMulFromTensor(input, upTensorE, ffnBuf.AsSpan(gateCols, upCols), upCols);
            _backend.Mul(ffnBuf.AsSpan(0, gateCols), ffnBuf.AsSpan(gateCols, upCols), ffnBuf.AsSpan(0, gateCols));
            MatMulFromTensor(ffnBuf.AsSpan(0, gateCols), downTensorE, _buffers.MoeExpertResultBuf.AsSpan(0, hidden), hidden);

            // Scale by routing weight and accumulate
            _backend.Scale(_buffers.MoeExpertResultBuf.AsSpan(0, hidden), weight);
            _backend.Add(_buffers.MoeAccumulatorBuf.AsSpan(0, hidden), _buffers.MoeExpertResultBuf.AsSpan(0, hidden));
        }

        // Copy accumulator to result
        _buffers.MoeAccumulatorBuf.AsSpan(0, hidden).CopyTo(resultBuf);
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
            var bias = biasInfo is not null ? _model.GetDequantizedWeights(biasInfo.Name) : null;
            _backend.LayerNorm(input, weights, bias ?? ReadOnlySpan<float>.Empty, outSpan, _cfg.NormEpsilon);
        }
    }

    private void ApplyPostNorm(Span<float> hiddenState, float[] tempBuf, int layer, bool isAttnPostNorm)
    {
        var tensors = _tn.ResolveLayer(layer, _cfg);
        var postNormInfo = isAttnPostNorm ? tensors.PostAttentionNormWeight : tensors.PostFfnNormWeight;
        if (postNormInfo is null) return;

        var weights = _model.GetDequantizedWeights(postNormInfo.Name);
        hiddenState.CopyTo(tempBuf);
        _backend.RmsNorm(tempBuf, weights, hiddenState, _cfg.NormEpsilon);
    }

    private void ApplyFinalNorm(Span<float> hiddenState, float[] tempBuf)
    {
        var name = _model.TryGetTensor("output_norm.weight", out _) ? "output_norm.weight" : "token_embd_norm.weight";
        var weights = _model.GetDequantizedWeights(name);
        hiddenState.CopyTo(tempBuf);
        _backend.RmsNorm(tempBuf, weights, hiddenState, _cfg.NormEpsilon);
    }

    private float[] ComputeLogits(ReadOnlySpan<float> hiddenState, float[] logits)
    {
        Tensor outputTensor;
        if (_model.TryGetTensor("output.weight", out var outTensor) && outTensor is not null)
            outputTensor = outTensor;
        else
            outputTensor = _model.GetTensor("token_embd.weight");

        if (outputTensor.ElementType == GgmlType.F32)
        {
            var weightData = MemoryMarshal.Cast<byte, float>(outputTensor.Data.Span);
            _backend.MatMulF32(hiddenState, weightData, logits, _cfg.HiddenSize, _cfg.VocabSize, outputTensor.Name);
        }
        else
        {
            _backend.MatMul(hiddenState, outputTensor.Data.Span, logits, outputTensor.ElementType, _cfg.HiddenSize, _cfg.VocabSize, outputTensor.Name);
        }

        return logits;
    }

    private void MatMulFromTensor(ReadOnlySpan<float> input, Tensor weight, Span<float> output, int cols)
    {
        var inDim = weight.ColumnCount;

        if (weight.ElementType == GgmlType.F32)
        {
            var weightData = MemoryMarshal.Cast<byte, float>(weight.Data.Span);
            _backend.MatMulF32(input[..inDim], weightData, output, inDim, cols, weight.Name);
        }
        else
        {
            _backend.MatMul(input[..inDim], weight.Data.Span, output, weight.ElementType, inDim, cols, weight.Name);
        }
    }

}