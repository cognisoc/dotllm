using Llmdot.Inference;
using Llmdot.Models;
using Xunit;

namespace Llmdot.Core.Tests.Inference;

public class ModelInfoTests
{
    [Fact]
    public void ModelInfo_SmolLm2_FieldsMatchConfig()
    {
        var config = new TransformerConfig
        {
            Architecture = "llama",
            Template = ExecutionTemplate.LlamaLike,
            HiddenSize = 576,
            LayerCount = 30,
            ContextLength = 2048,
            VocabSize = 49152,
            FfnDim = 1536,
            AttentionType = AttentionType.GQA,
            HeadCount = 9,
            HeadCountKv = 3,
            HeadDim = 64,
            FfnType = FfnType.SwiGLU,
            NormType = NormType.RmsNorm,
            TiedEmbeddings = true,
            RopeFreqBase = 10000f,
        };

        var info = ModelInfo.FromConfig(config);

        Assert.Equal("llama", info.Architecture);
        Assert.Equal("GQA", info.AttentionType);
        Assert.True(info.TiedEmbeddings);
        Assert.False(info.HasConvLayers);
        Assert.False(info.HasMoE);
    }

    [Fact]
    public void ModelInfo_Lfm2_HasConvLayers()
    {
        var config = new TransformerConfig
        {
            Architecture = "lfm2",
            HasConvLayers = true,
            ExpertCount = 0,
        };

        var info = ModelInfo.FromConfig(config);
        Assert.True(info.HasConvLayers);
        Assert.False(info.HasMoE);
    }

    [Fact]
    public void ModelInfo_Lfm2Moe_HasMoE()
    {
        var config = new TransformerConfig
        {
            Architecture = "lfm2_moe",
            HasConvLayers = true,
            ExpertCount = 8,
            ExpertUsedCount = 2,
        };

        var info = ModelInfo.FromConfig(config);
        Assert.True(info.HasMoE);
        Assert.Equal(8, info.ExpertCount);
    }
}
