using Llmdot.Inference;
using Xunit;

namespace Llmdot.Core.Tests.Inference;

public class GenerationOptionsBuilderTests
{
    [Fact]
    public void Build_Defaults_ReturnsReasonableValues()
    {
        var options = new GenerationOptionsBuilder().Build();
        Assert.Equal(256, options.MaxTokens);
        Assert.Equal(0.8f, options.Sampling.Temperature);
        Assert.Equal(40, options.Sampling.TopK);
        Assert.Equal(0.95f, options.Sampling.TopP);
    }

    [Fact]
    public void WithMaxTokens_SetsValue()
    {
        var options = new GenerationOptionsBuilder()
            .WithMaxTokens(512)
            .Build();
        Assert.Equal(512, options.MaxTokens);
    }

    [Fact]
    public void WithMaxTokens_Zero_Throws()
    {
        var builder = new GenerationOptionsBuilder();
        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithMaxTokens(0));
    }

    [Fact]
    public void WithTemperature_Zero_CreatesGreedyOptions()
    {
        var options = new GenerationOptionsBuilder()
            .WithTemperature(0f)
            .Build();
        Assert.Equal(0f, options.Sampling.Temperature);
    }

    [Fact]
    public void WithStopSequence_Accumulates()
    {
        var options = new GenerationOptionsBuilder()
            .WithStopSequence("<|end|>")
            .WithStopSequence("</s>")
            .Build();
        Assert.Equal(2, options.StopSequences.Length);
        Assert.Contains("<|end|>", options.StopSequences);
        Assert.Contains("</s>", options.StopSequences);
    }

    [Fact]
    public void WithSeed_SetsOnSamplingOptions()
    {
        var options = new GenerationOptionsBuilder()
            .WithSeed(42)
            .Build();
        Assert.Equal(42, options.Sampling.Seed);
    }

    [Fact]
    public void WithTopP_OutOfRange_Throws()
    {
        var builder = new GenerationOptionsBuilder();
        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithTopP(1.5f));
    }
}
