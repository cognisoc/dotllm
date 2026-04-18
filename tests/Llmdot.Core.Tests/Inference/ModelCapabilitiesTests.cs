using Llmdot.Models;
using Xunit;

namespace Llmdot.Core.Tests.Inference;

public class ModelCapabilitiesTests
{
    [Fact]
    public void FromConfig_MapsAllProperties()
    {
        var config = new TransformerConfig
        {
            Architecture = "llama",
            Template = ExecutionTemplate.LlamaLike,
            HiddenSize = 4096,
            LayerCount = 32,
            ContextLength = 4096,
            VocabSize = 32000,
            AttentionType = AttentionType.GQA,
            NormType = NormType.RmsNorm,
            FfnType = FfnType.SwiGLU,
            HasConvLayers = false,
            ExpertCount = 0,
            SlidingWindow = 0,
            AttnLogitSoftcap = null,
            FinalLogitSoftcap = null,
        };

        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);

        Assert.Equal("llama", caps.Architecture);
        Assert.Equal(ExecutionTemplate.LlamaLike, caps.Template);
        Assert.Equal(4096, caps.HiddenSize);
        Assert.Equal(32, caps.LayerCount);
        Assert.Equal(4096, caps.ContextLength);
        Assert.Equal(32000, caps.VocabSize);
        Assert.Equal(AttentionType.GQA, caps.AttentionType);
        Assert.Equal(NormType.RmsNorm, caps.NormType);
        Assert.Equal(FfnType.SwiGLU, caps.FfnType);
        Assert.False(caps.HasConvLayers);
    }

    [Fact]
    public void HasMoE_TrueWhenExpertCountGreaterThanZero()
    {
        var config = CreateConfig(expertCount: 8);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.True(caps.HasMoE);
    }

    [Fact]
    public void HasMoE_FalseWhenExpertCountIsZero()
    {
        var config = CreateConfig(expertCount: 0);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.False(caps.HasMoE);
    }

    [Fact]
    public void HasSlidingWindow_TrueWhenSet()
    {
        var config = CreateConfig(slidingWindow: 4096);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.True(caps.HasSlidingWindow);
    }

    [Fact]
    public void HasSlidingWindow_FalseWhenZero()
    {
        var config = CreateConfig(slidingWindow: 0);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.False(caps.HasSlidingWindow);
    }

    [Fact]
    public void HasSoftcapping_TrueWhenAttnLogitSoftcapSet()
    {
        var config = CreateConfig(attnLogitSoftcap: 50.0f);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.True(caps.HasSoftcapping);
    }

    [Fact]
    public void HasSoftcapping_TrueWhenFinalLogitSoftcapSet()
    {
        var config = CreateConfig(finalLogitSoftcap: 30.0f);
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.True(caps.HasSoftcapping);
    }

    [Fact]
    public void HasSoftcapping_FalseWhenNoSoftcaps()
    {
        var config = CreateConfig();
        var caps = Llmdot.Inference.ModelCapabilities.FromConfig(config);
        Assert.False(caps.HasSoftcapping);
    }

    private static TransformerConfig CreateConfig(
        int expertCount = 0,
        int slidingWindow = 0,
        float? attnLogitSoftcap = null,
        float? finalLogitSoftcap = null) => new()
    {
        Architecture = "llama",
        Template = ExecutionTemplate.LlamaLike,
        HiddenSize = 4096,
        LayerCount = 32,
        ContextLength = 4096,
        VocabSize = 32000,
        FfnDim = 11008,
        AttentionType = AttentionType.MHA,
        HeadCount = 32,
        HeadCountKv = 32,
        HeadDim = 128,
        NormType = NormType.RmsNorm,
        NormEpsilon = 1e-5f,
        FfnType = FfnType.SwiGLU,
        ExpertCount = expertCount,
        SlidingWindow = slidingWindow,
        AttnLogitSoftcap = attnLogitSoftcap,
        FinalLogitSoftcap = finalLogitSoftcap,
    };
}