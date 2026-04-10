using Dotllm.Models;

namespace Dotllm.Inference;

public sealed class ModelCapabilities
{
    public string Architecture { get; init; } = string.Empty;
    public ExecutionTemplate Template { get; init; }
    public int HiddenSize { get; init; }
    public int LayerCount { get; init; }
    public int ContextLength { get; init; }
    public int VocabSize { get; init; }
    public AttentionType AttentionType { get; init; }
    public NormType NormType { get; init; }
    public FfnType FfnType { get; init; }
    public bool HasConvLayers { get; init; }
    public bool HasMoE => ExpertCount > 0;
    public int ExpertCount { get; init; }
    public bool HasSlidingWindow => SlidingWindow > 0;
    public int SlidingWindow { get; init; }
    public bool HasSoftcapping => AttnLogitSoftcap.HasValue || FinalLogitSoftcap.HasValue;

    public float? AttnLogitSoftcap { get; init; }
    public float? FinalLogitSoftcap { get; init; }

    internal static ModelCapabilities FromConfig(TransformerConfig config) => new()
    {
        Architecture = config.Architecture,
        Template = config.Template,
        HiddenSize = config.HiddenSize,
        LayerCount = config.LayerCount,
        ContextLength = config.ContextLength,
        VocabSize = config.VocabSize,
        AttentionType = config.AttentionType,
        NormType = config.NormType,
        FfnType = config.FfnType,
        HasConvLayers = config.HasConvLayers,
        ExpertCount = config.ExpertCount,
        SlidingWindow = config.SlidingWindow,
        AttnLogitSoftcap = config.AttnLogitSoftcap,
        FinalLogitSoftcap = config.FinalLogitSoftcap,
    };
}