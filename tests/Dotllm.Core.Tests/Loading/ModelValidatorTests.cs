using Dotllm.Loading;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Loading;

public class ModelValidatorTests
{
    private static GgufModel CreateModel(params string[] tensorNames)
    {
        var infos = tensorNames.Select(n => new GgufTensorInfo
        {
            Name = n,
            Dimensions = [1],
            Type = GgmlType.F32,
            Offset = 0,
            ElementCount = 1,
        }).ToList();
        return new GgufModel
        {
            Version = 3,
            TensorCount = (ulong)infos.Count,
            Metadata = new GgufMetadata(new Dictionary<string, GgufMetadataValue>()),
            TensorInfos = infos,
            TensorDataOffset = 0,
        };
    }

    [Fact]
    public void Validate_MinimalValidModel_ReturnsNoErrors()
    {
        var model = CreateModel(
            "token_embd.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
            "blk.0.attn_output.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight");
        var config = new TransformerConfig
        {
            LayerCount = 1,
            LayerTypes = [LayerType.Attention],
        };

        var result = ModelValidator.Validate(model, config);
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void Validate_MissingEmbedding_ReturnsError()
    {
        var model = CreateModel("output_norm.weight");
        var config = new TransformerConfig { LayerCount = 0, LayerTypes = [] };

        var result = ModelValidator.Validate(model, config);
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("token_embd.weight"));
    }

    [Fact]
    public void Validate_MissingAttentionWeights_ReturnsError()
    {
        var model = CreateModel("token_embd.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight");
        var config = new TransformerConfig
        {
            LayerCount = 1,
            LayerTypes = [LayerType.Attention],
        };

        var result = ModelValidator.Validate(model, config);
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("attention weights"));
    }

    [Fact]
    public void Validate_TiedEmbeddings_ReturnsWarning()
    {
        var model = CreateModel("token_embd.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight");
        var config = new TransformerConfig
        {
            LayerCount = 1,
            LayerTypes = [LayerType.Attention],
            TiedEmbeddings = true,
        };

        var result = ModelValidator.Validate(model, config);
        Assert.Contains(result.Warnings, w => w.Contains("tied embeddings"));
    }

    [Fact]
    public void Validate_MoEWithoutGateTensor_ReturnsWarning()
    {
        var model = CreateModel("token_embd.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_up.weight", "blk.0.ffn_down.weight");
        var config = new TransformerConfig
        {
            LayerCount = 1,
            LayerTypes = [LayerType.Attention],
            ExpertCount = 8,
        };

        var result = ModelValidator.Validate(model, config);
        Assert.Contains(result.Warnings, w => w.Contains("ffn_gate_inp.weight"));
    }

    [Fact]
    public void Validate_ConvLayerMissingConvWeight_ReturnsError()
    {
        var model = CreateModel("token_embd.weight", "output_norm.weight",
            "blk.0.attn_norm.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight");
        var config = new TransformerConfig
        {
            LayerCount = 1,
            LayerTypes = [LayerType.Conv],
            HasConvLayers = true,
        };

        var result = ModelValidator.Validate(model, config);
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("conv weight"));
    }
}
