using Dotllm.Models;

namespace Dotllm.Loading;

internal sealed class TensorNameResolver
{
    private readonly Dictionary<string, GgufTensorInfo> _tensors;

    public TensorNameResolver(IReadOnlyList<GgufTensorInfo> tensorInfos)
    {
        _tensors = new Dictionary<string, GgufTensorInfo>(tensorInfos.Count, StringComparer.OrdinalIgnoreCase);
        foreach (var t in tensorInfos)
            _tensors[t.Name] = t;
    }

    public GgufTensorInfo? TryGet(string name) =>
        _tensors.TryGetValue(name, out var info) ? info : null;

    public GgufTensorInfo Get(string name) =>
        TryGet(name) ?? throw new KeyNotFoundException($"Tensor '{name}' not found in model.");

    public GgufTensorInfo TokenEmbeddings => Get("token_embd.weight");

    public GgufTensorInfo OutputNorm => Get("output_norm.weight");

    public GgufTensorInfo? TryOutputProjection =>
        TryGet("output.weight");

    public GgufTensorInfo LayerNormWeight(int layer) =>
        Get($"blk.{layer}.attn_norm.weight");

    public GgufTensorInfo? TryLayerNormBias(int layer) =>
        TryGet($"blk.{layer}.attn_norm.bias");

    public GgufTensorInfo FfnUpWeight(int layer) =>
        Get($"blk.{layer}.ffn_up.weight");

    public GgufTensorInfo? TryFfnUpBias(int layer) =>
        TryGet($"blk.{layer}.ffn_up.bias");

    public GgufTensorInfo? TryFfnGateWeight(int layer) =>
        TryGet($"blk.{layer}.ffn_gate.weight");

    public GgufTensorInfo? TryFfnGateBias(int layer) =>
        TryGet($"blk.{layer}.ffn_gate.bias");

    public GgufTensorInfo FfnDownWeight(int layer) =>
        Get($"blk.{layer}.ffn_down.weight");

    public GgufTensorInfo? TryFfnDownBias(int layer) =>
        TryGet($"blk.{layer}.ffn_down.bias");

    public GgufTensorInfo? TryFfnNormWeight(int layer) =>
        TryGet($"blk.{layer}.ffn_norm.weight");

    public (GgufTensorInfo q, GgufTensorInfo k, GgufTensorInfo v) GetSeparateQkv(int layer) =>
        (Get($"blk.{layer}.attn_q.weight"),
         Get($"blk.{layer}.attn_k.weight"),
         Get($"blk.{layer}.attn_v.weight"));

    public GgufTensorInfo? TryFusedQkvWeight(int layer) =>
        TryGet($"blk.{layer}.attn_qkv.weight");

    public GgufTensorInfo AttentionOutputWeight(int layer) =>
        Get($"blk.{layer}.attn_output.weight");

    public GgufTensorInfo? TryAttentionOutputBias(int layer) =>
        TryGet($"blk.{layer}.attn_output.bias");

    public GgufTensorInfo? TryPostAttentionNormWeight(int layer) =>
        TryGet($"blk.{layer}.post_attention_norm.weight");

    public GgufTensorInfo? TryPostFfnNormWeight(int layer) =>
        TryGet($"blk.{layer}.post_ffw_norm.weight");

    public GgufTensorInfo? TryLayerNorm2Weight(int layer) =>
        TryGet($"blk.{layer}.attn_norm_2.weight") ?? TryGet($"blk.{layer}.ffn_norm.weight");

    public GgufTensorInfo? TryConvWeight(int layer) =>
        TryGet($"blk.{layer}.conv1d.weight") ?? TryGet($"blk.{layer}.shortconv.conv.weight");

    public GgufTensorInfo? TryConvBias(int layer) =>
        TryGet($"blk.{layer}.conv1d.bias") ?? TryGet($"blk.{layer}.shortconv.conv.bias");

    public GgufTensorInfo? TryConvInProj(int layer) =>
        TryGet($"blk.{layer}.shortconv.in_proj.weight");

    public GgufTensorInfo? TryConvOutProj(int layer) =>
        TryGet($"blk.{layer}.shortconv.out_proj.weight");

    public GgufTensorInfo? TryAttentionQNorm(int layer) =>
        TryGet($"blk.{layer}.attn_q_norm.weight");

    public GgufTensorInfo? TryAttentionKNorm(int layer) =>
        TryGet($"blk.{layer}.attn_k_norm.weight");

    public ResolvedLayerTensors ResolveLayer(int layer, TransformerConfig config)
    {
        var normWeight = LayerNormWeight(layer);
        var normBias = TryLayerNormBias(layer);
        var ffnNormWeight = TryFfnNormWeight(layer);

        var isAttention = config.LayerTypes.Length > layer && config.LayerTypes[layer] == LayerType.Attention;

        (GgufTensorInfo q, GgufTensorInfo k, GgufTensorInfo v) qkvTensors = default;
        GgufTensorInfo? fusedQkvWeight = null;
        GgufTensorInfo? attnOutputWeight = null;

        if (isAttention)
        {
            if (config.QkvLayout == QkvLayout.Fused)
            {
                fusedQkvWeight = TryFusedQkvWeight(layer)
                    ?? throw new KeyNotFoundException($"Expected fused QKV tensor 'blk.{layer}.attn_qkv.weight' for QkvLayout.Fused.");
            }
            else
            {
                qkvTensors = GetSeparateQkv(layer);
            }

            attnOutputWeight = AttentionOutputWeight(layer);
        }

        var ffnUpWeight = FfnUpWeight(layer);
        var ffnDownWeight = FfnDownWeight(layer);
        var ffnGateWeight = TryFfnGateWeight(layer);

        return new ResolvedLayerTensors
        {
            Layer = layer,
            NormWeight = normWeight,
            NormBias = normBias,
            FfnNormWeight = ffnNormWeight,
            Q = qkvTensors.q,
            K = qkvTensors.k,
            V = qkvTensors.v,
            FusedQkvWeight = fusedQkvWeight,
            AttnOutputWeight = attnOutputWeight,
            FfnUpWeight = ffnUpWeight,
            FfnDownWeight = ffnDownWeight,
            FfnGateWeight = ffnGateWeight,
            PostAttentionNormWeight = TryPostAttentionNormWeight(layer),
            PostFfnNormWeight = TryPostFfnNormWeight(layer),
            ConvWeight = TryConvWeight(layer),
            ConvBias = TryConvBias(layer),
            ConvInProj = TryConvInProj(layer),
            ConvOutProj = TryConvOutProj(layer),
            AttentionQNorm = TryAttentionQNorm(layer),
            AttentionKNorm = TryAttentionKNorm(layer),
        };
    }
}

internal sealed class ResolvedLayerTensors
{
    public int Layer { get; init; }
    public GgufTensorInfo NormWeight { get; init; } = null!;
    public GgufTensorInfo? NormBias { get; init; }
    public GgufTensorInfo? FfnNormWeight { get; init; }
    public GgufTensorInfo? Q { get; init; }
    public GgufTensorInfo? K { get; init; }
    public GgufTensorInfo? V { get; init; }
    public GgufTensorInfo? FusedQkvWeight { get; init; }
    public GgufTensorInfo? AttnOutputWeight { get; init; }
    public GgufTensorInfo FfnUpWeight { get; init; } = null!;
    public GgufTensorInfo FfnDownWeight { get; init; } = null!;
    public GgufTensorInfo? FfnGateWeight { get; init; }
    public GgufTensorInfo? PostAttentionNormWeight { get; init; }
    public GgufTensorInfo? PostFfnNormWeight { get; init; }
    public GgufTensorInfo? ConvWeight { get; init; }
    public GgufTensorInfo? ConvBias { get; init; }
    public GgufTensorInfo? ConvInProj { get; init; }
    public GgufTensorInfo? ConvOutProj { get; init; }
    public GgufTensorInfo? AttentionQNorm { get; init; }
    public GgufTensorInfo? AttentionKNorm { get; init; }
}