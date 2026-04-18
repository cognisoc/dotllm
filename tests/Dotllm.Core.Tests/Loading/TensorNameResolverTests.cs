using Dotllm.Loading;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Loading;

public class TensorNameResolverTests
{
    private static TensorNameResolver CreateResolver(params string[] tensorNames)
    {
        var infos = tensorNames.Select(n => new GgufTensorInfo
        {
            Name = n,
            Dimensions = [1],
            Type = GgmlType.F32,
            Offset = 0,
            ElementCount = 1,
        }).ToList();
        return new TensorNameResolver(infos);
    }

    [Fact]
    public void TokenEmbeddings_ResolvesCorrectly()
    {
        var resolver = CreateResolver("token_embd.weight");
        var result = resolver.TokenEmbeddings;
        Assert.Equal("token_embd.weight", result.Name);
    }

    [Fact]
    public void TryOutputProjection_ReturnsNull_WhenTiedEmbeddings()
    {
        var resolver = CreateResolver("token_embd.weight");
        var result = resolver.TryOutputProjection;
        Assert.Null(result);
    }

    [Fact]
    public void TryOutputProjection_ReturnsTensor_WhenPresent()
    {
        var resolver = CreateResolver("token_embd.weight", "output.weight");
        var result = resolver.TryOutputProjection;
        Assert.NotNull(result);
        Assert.Equal("output.weight", result!.Name);
    }

    [Fact]
    public void GetSeparateQkv_ResolvesLlamaPatterns()
    {
        var resolver = CreateResolver(
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight");
        var (q, k, v) = resolver.GetSeparateQkv(0);
        Assert.Equal("blk.0.attn_q.weight", q.Name);
        Assert.Equal("blk.0.attn_k.weight", k.Name);
        Assert.Equal("blk.0.attn_v.weight", v.Name);
    }

    [Fact]
    public void TryConvWeight_ResolvesShortconvPattern()
    {
        var resolver = CreateResolver("blk.0.shortconv.conv.weight");
        var result = resolver.TryConvWeight(0);
        Assert.NotNull(result);
        Assert.Equal("blk.0.shortconv.conv.weight", result!.Name);
    }

    [Fact]
    public void TryConvWeight_ResolvesConv1dPattern()
    {
        var resolver = CreateResolver("blk.0.conv1d.weight");
        var result = resolver.TryConvWeight(0);
        Assert.NotNull(result);
        Assert.Equal("blk.0.conv1d.weight", result!.Name);
    }

    [Fact]
    public void Get_MissingTensor_Throws()
    {
        var resolver = CreateResolver("token_embd.weight");
        Assert.Throws<KeyNotFoundException>(() => resolver.Get("nonexistent.weight"));
    }
}
