using Dotllm.Models;

namespace Dotllm.Inference;

public sealed class ModelInfo
{
    public string Architecture { get; init; } = string.Empty;
    public string Template { get; init; } = string.Empty;
    public int HiddenSize { get; init; }
    public int LayerCount { get; init; }
    public int ContextLength { get; init; }
    public int VocabSize { get; init; }
    public int FfnDim { get; init; }
    public string AttentionType { get; init; } = string.Empty;
    public int HeadCount { get; init; }
    public int HeadCountKv { get; init; }
    public int HeadDim { get; init; }
    public string NormType { get; init; } = string.Empty;
    public string FfnType { get; init; } = string.Empty;
    public bool TiedEmbeddings { get; init; }
    public bool HasConvLayers { get; init; }
    public bool HasMoE { get; init; }
    public int ExpertCount { get; init; }
    public float RopeFreqBase { get; init; }

    public static ModelInfo FromConfig(TransformerConfig config) => new()
    {
        Architecture = config.Architecture,
        Template = config.Template.ToString(),
        HiddenSize = config.HiddenSize,
        LayerCount = config.LayerCount,
        ContextLength = config.ContextLength,
        VocabSize = config.VocabSize,
        FfnDim = config.FfnDim,
        AttentionType = config.AttentionType.ToString(),
        HeadCount = config.HeadCount,
        HeadCountKv = config.HeadCountKv,
        HeadDim = config.HeadDim,
        NormType = config.NormType.ToString(),
        FfnType = config.FfnType.ToString(),
        TiedEmbeddings = config.TiedEmbeddings,
        HasConvLayers = config.HasConvLayers,
        HasMoE = config.ExpertCount > 0,
        ExpertCount = config.ExpertCount,
        RopeFreqBase = config.RopeFreqBase,
    };
}
