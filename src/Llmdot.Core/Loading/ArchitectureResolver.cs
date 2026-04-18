using System.Globalization;
using Llmdot.Models;

namespace Llmdot.Loading;

internal static class ArchitectureResolver
{
    private static readonly HashSet<string> SupportedArchitectures = new(StringComparer.OrdinalIgnoreCase)
    {
        "llama", "phi3", "qwen2", "phi2", "gptneox", "gemma", "gemma2", "stablelm",
        "lfm2", "lfm2_moe",
    };

    public static TransformerConfig Resolve(GgufModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var arch = model.Metadata.TryGetValue("general.architecture", out var archVal) && archVal is not null
            ? archVal.Value?.ToString() ?? string.Empty
            : string.Empty;

        if (string.IsNullOrWhiteSpace(arch))
            throw new InvalidDataException("GGUF model missing 'general.architecture' metadata key.");

        if (!SupportedArchitectures.Contains(arch))
            throw new InvalidDataException($"Unsupported GGUF architecture: '{arch}'. Supported: {string.Join(", ", SupportedArchitectures)}");

        var template = ResolveTemplate(arch);
        var meta = model.Metadata;

        var hiddenSize = GetInt(meta, arch, "embedding_length");
        var layerCount = GetInt(meta, arch, "block_count");
        var contextLength = GetInt(meta, arch, "context_length");
        var ffnDim = GetInt(meta, arch, "feed_forward_length");
        var headCount = GetInt(meta, arch, "attention.head_count");
        var headCountKvRaw = GetOrDefaultArray(meta, arch, "attention.head_count_kv");
        var headCountKv = headCountKvRaw is { Length: > 0 }
            ? headCountKvRaw.Max()
            : GetOrDefaultInt(meta, arch, "attention.head_count_kv", headCount);
        var headDim = GetOrDefaultInt(meta, arch, "attention.key_length", hiddenSize / Math.Max(headCount, 1));
        var vocabSize = ResolveVocabSize(meta);

        var (normType, normEpsilon) = ResolveNormType(meta, arch);

        var ropeFreqBase = GetOrDefaultFloat(meta, arch, "rope.freq_base", 10000f);
        var ropeDimCount = GetOrDefaultInt(meta, arch, "rope.dimension_count", headDim);
        var (ropeScalingType, ropeScalingFactor) = ResolveRopeScaling(meta, arch);

        var attentionType = DetermineAttentionType(headCount, headCountKv);
        var qkvLayout = DetermineQkvLayout(model, arch);
        var ffnType = DetermineFfnType(model, arch, template);
        var parallelResidual = GetOrDefaultBool(meta, arch, "use_parallel_residual", template == ExecutionTemplate.GptNeoXLike);
        var tiedEmbeddings = !model.TensorInfos.Any(t => t.Name == "output.weight");
        var embeddingScale = ResolveEmbeddingScale(meta, arch, hiddenSize, template);
        var hasPostNorm = model.TensorInfos.Any(t => t.Name.Contains("post_attention_norm") || t.Name.Contains("post_ffw_norm"));

        var slidingWindow = GetOrDefaultInt(meta, arch, "attention.sliding_window", 0);
        var attnSoftcap = GetNullableFloat(meta, arch, "attn_logit_softcapping");
        var finalSoftcap = GetNullableFloat(meta, arch, "final_logit_softcapping");

        var expertCount = GetOrDefaultInt(meta, arch, "expert_count", 0);
        var expertUsedCount = GetOrDefaultInt(meta, arch, "expert_used_count", 0);
        var expertFfnDim = GetOrDefaultInt(meta, arch, "expert_feed_forward_length", 0);

        var hasConvLayers = template == ExecutionTemplate.Lfm2Like;
        var convKernelSize = hasConvLayers ? GetOrDefaultInt(meta, arch, "shortconv.l_cache", 3) : 0;
        var layerTypes = hasConvLayers
            ? ResolveLayerTypes(model, layerCount)
            : CreateAllAttentionLayers(layerCount);

        var bosTokenId = GetOrDefaultInt(meta, "tokenizer.ggml", "bos_token_id", 1);
        var eosTokenId = GetOrDefaultInt(meta, "tokenizer.ggml", "eos_token_id", 2);

        return new TransformerConfig
        {
            Architecture = arch,
            Template = template,
            HiddenSize = hiddenSize,
            LayerCount = layerCount,
            ContextLength = contextLength,
            VocabSize = vocabSize,
            FfnDim = ffnDim,
            AttentionType = attentionType,
            HeadCount = headCount,
            HeadCountKv = headCountKv,
            HeadCountKvPerLayer = headCountKvRaw,
            HeadDim = headDim,
            QkvLayout = qkvLayout,
            NormType = normType,
            NormEpsilon = normEpsilon,
            HasPostNorm = hasPostNorm,
            RopeFreqBase = ropeFreqBase,
            RopeDimensionCount = ropeDimCount,
            RopeScalingType = ropeScalingType,
            RopeScalingFactor = ropeScalingFactor,
            FfnType = ffnType,
            ParallelResidual = parallelResidual,
            TiedEmbeddings = tiedEmbeddings,
            EmbeddingScale = embeddingScale,
            SlidingWindow = slidingWindow,
            AttnLogitSoftcap = attnSoftcap,
            FinalLogitSoftcap = finalSoftcap,
            ExpertCount = expertCount,
            ExpertUsedCount = expertUsedCount,
            ExpertFfnDim = expertFfnDim,
            HasConvLayers = hasConvLayers,
            ConvKernelSize = convKernelSize,
            LayerTypes = layerTypes,
            BosTokenId = bosTokenId,
            EosTokenId = eosTokenId,
        };
    }

    private static ExecutionTemplate ResolveTemplate(string arch) => arch.ToLowerInvariant() switch
    {
        "llama" or "phi3" or "qwen2" or "stablelm" => ExecutionTemplate.LlamaLike,
        "phi2" or "gptneox" => ExecutionTemplate.GptNeoXLike,
        "gemma" or "gemma2" => ExecutionTemplate.GemmaLike,
        "lfm2" or "lfm2_moe" => ExecutionTemplate.Lfm2Like,
        _ => ExecutionTemplate.LlamaLike,
    };

    internal static AttentionType DetermineAttentionType(int headCount, int headCountKv)
    {
        if (headCount == headCountKv) return AttentionType.MHA;
        if (headCountKv == 1) return AttentionType.MQA;
        return AttentionType.GQA;
    }

    private static (NormType type, float epsilon) ResolveNormType(GgufMetadata meta, string arch)
    {
        var rmsKey = $"{arch}.attention.layer_norm_rms_epsilon";
        var lnKey = $"{arch}.attention.layer_norm_epsilon";

        if (meta.TryGetValue(rmsKey, out var rmsVal) && rmsVal is not null)
            return (NormType.RmsNorm, Convert.ToSingle(rmsVal.Value, CultureInfo.InvariantCulture));
        if (meta.TryGetValue(lnKey, out var lnVal) && lnVal is not null)
            return (NormType.LayerNorm, Convert.ToSingle(lnVal.Value, CultureInfo.InvariantCulture));

        throw new InvalidDataException($"No norm epsilon found for architecture '{arch}'. Expected '{rmsKey}' or '{lnKey}'.");
    }

    private static QkvLayout DetermineQkvLayout(GgufModel model, string arch)
    {
        var hasFusedQkv = model.TensorInfos.Any(t => t.Name.Contains("attn_qkv"));
        return hasFusedQkv ? QkvLayout.Fused : QkvLayout.Separate;
    }

    private static FfnType DetermineFfnType(GgufModel model, string arch, ExecutionTemplate template)
    {
        var hasGate = model.TensorInfos.Any(t => t.Name.Contains("ffn_gate"));
        if (!hasGate)
            return FfnType.Standard;

        return template == ExecutionTemplate.GemmaLike ? FfnType.GeGLU : FfnType.SwiGLU;
    }

    private static float ResolveEmbeddingScale(GgufMetadata meta, string arch, int hiddenSize, ExecutionTemplate template)
    {
        var scale = GetNullableFloat(meta, arch, "embedding_scale");
        if (scale.HasValue)
            return scale.Value;

        return template == ExecutionTemplate.GemmaLike
            ? MathF.Sqrt(hiddenSize)
            : 1f;
    }

    private static (RopeScalingType type, float factor) ResolveRopeScaling(GgufMetadata meta, string arch)
    {
        var scalingTypeStr = meta.GetOrDefault($"{arch}.rope.scaling.type", string.Empty)?.ToString() ?? string.Empty;
        var factor = GetOrDefaultFloat(meta, arch, "rope.scaling.factor", 1f);

        var scalingType = scalingTypeStr.ToLowerInvariant() switch
        {
            "linear" => RopeScalingType.Linear,
            "yarn" => RopeScalingType.YaRN,
            "longrope" => RopeScalingType.LongRoPE,
            _ => RopeScalingType.None,
        };

        return (scalingType, factor);
    }

    private static int ResolveVocabSize(GgufMetadata meta)
    {
        if (meta.TryGetValue("tokenizer.ggml.tokens", out var val) && val is not null && val.Value is object[] arr)
            return arr.Length;
        return 0;
    }

    private static int GetInt(GgufMetadata meta, string arch, string key)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null)
            return Convert.ToInt32(val.Value, CultureInfo.InvariantCulture);
        return 0;
    }

    private static int GetOrDefaultInt(GgufMetadata meta, string arch, string key, int defaultValue)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null)
            return Convert.ToInt32(val.Value, CultureInfo.InvariantCulture);
        return defaultValue;
    }

    private static float GetOrDefaultFloat(GgufMetadata meta, string arch, string key, float defaultValue)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null)
            return Convert.ToSingle(val.Value, CultureInfo.InvariantCulture);
        return defaultValue;
    }

    private static float? GetNullableFloat(GgufMetadata meta, string arch, string key)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null)
            return Convert.ToSingle(val.Value, CultureInfo.InvariantCulture);
        return null;
    }

    private static bool GetOrDefaultBool(GgufMetadata meta, string arch, string key, bool defaultValue)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null)
            return Convert.ToBoolean(val.Value, CultureInfo.InvariantCulture);
        return defaultValue;
    }

    private static int[] GetOrDefaultArray(GgufMetadata meta, string arch, string key)
    {
        if (meta.TryGetValue($"{arch}.{key}", out var val) && val is not null && val.Value is object[] arr)
        {
            var result = new int[arr.Length];
            for (var i = 0; i < arr.Length; i++)
                result[i] = Convert.ToInt32(arr[i], CultureInfo.InvariantCulture);
            return result;
        }
        return [];
    }

    private static LayerType[] ResolveLayerTypes(GgufModel model, int layerCount)
    {
        var types = new LayerType[layerCount];
        for (var i = 0; i < layerCount; i++)
        {
            var hasAttentionWeights = model.TensorInfos.Any(t =>
                t.Name.Equals($"blk.{i}.attn_q.weight", StringComparison.OrdinalIgnoreCase) ||
                t.Name.Equals($"blk.{i}.attn_k.weight", StringComparison.OrdinalIgnoreCase) ||
                t.Name.Equals($"blk.{i}.attn_qkv.weight", StringComparison.OrdinalIgnoreCase));
            types[i] = hasAttentionWeights ? LayerType.Attention : LayerType.Conv;
        }
        return types;
    }

    private static LayerType[] CreateAllAttentionLayers(int layerCount)
    {
        var types = new LayerType[layerCount];
        Array.Fill(types, LayerType.Attention);
        return types;
    }
}