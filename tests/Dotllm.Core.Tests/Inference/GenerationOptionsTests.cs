using Dotllm.Inference;
using Dotllm.Sampling;
using Xunit;

namespace Dotllm.Core.Tests.Inference;

public class GenerationOptionsTests
{
    [Fact]
    public void Defaults_AreSet()
    {
        var options = new GenerationOptions();

        Assert.Equal(256, options.MaxTokens);
        Assert.NotNull(options.Sampling);
        Assert.Null(options.StopSequence);
        Assert.Empty(options.StopSequences);
    }

    [Fact]
    public void StopSequences_CanBeSet()
    {
        var options = new GenerationOptions
        {
            StopSequences = ["</s>", "\n\n"],
        };

        Assert.Equal(2, options.StopSequences.Length);
        Assert.Contains("</s>", options.StopSequences);
        Assert.Contains("\n\n", options.StopSequences);
    }

    [Fact]
    public void StopSequences_DefaultsToEmptyArray()
    {
        var options = new GenerationOptions();
        Assert.NotNull(options.StopSequences);
        Assert.Empty(options.StopSequences);
    }

    [Fact]
    public void SamplingDefaults_AreSet()
    {
        var sampling = new SamplingOptions();

        Assert.Equal(40, sampling.TopK);
        Assert.Equal(0.95f, sampling.TopP);
        Assert.Equal(0.8f, sampling.Temperature);
        Assert.Equal(1.1f, sampling.RepeatPenalty);
        Assert.Equal(64, sampling.RepeatPenaltyWindowSize);
        Assert.Equal(-1, sampling.Seed);
    }
}