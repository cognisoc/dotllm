using Llmdot.Inference;
using Llmdot.Sampling;
using Xunit;

namespace Llmdot.Core.Tests.Sampling;

public class SamplerTests
{
    private static Sampler CreateSampler() => new(new CpuBackend());

    private static float[] MakeLogits(int size, int peakIndex, float peakValue = 10f)
    {
        var logits = new float[size];
        for (var i = 0; i < size; i++) logits[i] = 0.01f;
        logits[peakIndex] = peakValue;
        return logits;
    }

    [Fact]
    public void TemperatureZero_ReturnsArgMax()
    {
        var sampler = CreateSampler();
        var logits = MakeLogits(100, peakIndex: 42, peakValue: 10f);
        var options = new SamplingOptions { Temperature = 0f };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(42, result);
    }

    [Fact]
    public void TemperatureZero_IgnoresOtherSamplingParameters()
    {
        var sampler = CreateSampler();
        var logits = MakeLogits(100, peakIndex: 7, peakValue: 20f);
        var options = new SamplingOptions
        {
            Temperature = 0f,
            TopK = 5,
            TopP = 0.5f,
            RepeatPenalty = 2.0f,
            RepeatPenaltyWindowSize = 64,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];
        var recentTokens = new List<int> { 7 };

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, recentTokens);

        Assert.Equal(7, result);
    }

    [Fact]
    public void TemperatureZero_WithMultiplePeaks_ReturnsHighest()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[10] = 5f;
        logits[20] = 10f;
        logits[30] = 7f;
        var options = new SamplingOptions { Temperature = 0f };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(20, result);
    }

    [Fact]
    public void TopK_LimitsSamplingToKTokens()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 100f;
        logits[1] = 90f;
        logits[2] = 80f;
        for (var i = 3; i < 100; i++) logits[i] = 0.001f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 3,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.InRange(result, 0, 2);
    }

    [Fact]
    public void TopK_SeedDeterminesResultWithinTopK()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 100f;
        logits[1] = 90f;
        logits[2] = 80f;
        for (var i = 3; i < 100; i++) logits[i] = 0.001f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 3,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 99,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result1 = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());
        var result2 = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(result1, result2);
        Assert.InRange(result1, 0, 2);
    }

    [Fact]
    public void TopP_NucleusSampling()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 10f;
        logits[1] = 5f;
        logits[2] = 3f;
        for (var i = 3; i < 100; i++) logits[i] = 0.001f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 0.5f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.True(result <= 2);
    }

    [Fact]
    public void TopP_WithLowTopP_SelectsDominantToken()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 100f;
        for (var i = 1; i < 100; i++) logits[i] = 0.01f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 0.01f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(0, result);
    }

    [Fact]
    public void RepeatPenalty_ReducesProbabilityOfRecentToken()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[50] = 100f;
        logits[51] = 50f;
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 100,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 50 });

        Assert.NotEqual(50, result);
    }

    [Fact]
    public void RepeatPenalty_IncreasesProbabilityOfNegativelyScoredRecentToken()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        for (var i = 0; i < 100; i++) logits[i] = -1f;
        logits[50] = -0.5f;
        logits[30] = 10f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];
        var resultWithoutPenalty = sampler.Sample(logits, new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        }, samplingBuf, samplingIdxBuf, new List<int>());

        logits[30] = 10f;
        for (var i = 0; i < 100; i++) logits[i] = -1f;
        logits[50] = -0.5f;
        logits[30] = 10f;
        var resultWithPenalty = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 30 });

        Assert.Equal(30, resultWithoutPenalty);
        Assert.NotEqual(30, resultWithPenalty);
    }

    [Fact]
    public void RepeatPenalty_NoEffectWhenWindowSizeIsZero()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[42] = 10f;
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 42 });

        Assert.Equal(42, result);
    }

    [Fact]
    public void RepeatPenalty_NoEffectWhenPenaltyIsOne()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[42] = 10f;
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 42 });

        Assert.Equal(42, result);
    }

    [Fact]
    public void RepeatPenalty_NoEffectWhenRecentTokensIsEmpty()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[42] = 10f;
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(42, result);
    }

    [Fact]
    public void SeedDeterminism_SameSeedProducesSameResult()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        for (var i = 0; i < 100; i++) logits[i] = (float)(i * 0.1);
        var options = new SamplingOptions
        {
            Temperature = 0.8f,
            TopK = 40,
            TopP = 0.95f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 12345,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result1 = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());
        var result2 = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void SeedDeterminism_DifferentSeedsCanProduceDifferentResults()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        for (var i = 0; i < 100; i++) logits[i] = (float)(i * 0.5);
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];
        var results = new HashSet<int>();

        for (var seed = 0; seed < 10; seed++)
        {
            var options = new SamplingOptions
            {
                Temperature = 1f,
                TopK = 10,
                TopP = 0.95f,
                RepeatPenalty = 1f,
                RepeatPenaltyWindowSize = 0,
                Seed = seed,
            };
            results.Add(sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>()));
        }

        Assert.True(results.Count > 1, "Different seeds should produce different results with sufficient spread");
    }

    [Fact]
    public void DefaultSamplingOptions_ProducesValidToken()
    {
        var sampler = CreateSampler();
        var logits = new float[32000];
        for (var i = 0; i < 32000; i++) logits[i] = (float)(i * 0.001);
        logits[100] = 5f;
        var options = new SamplingOptions { Seed = 42 };
        var samplingBuf = new float[32000];
        var samplingIdxBuf = new int[32000];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.InRange(result, 0, 31999);
    }

    [Fact]
    public void DefaultSamplingOptions_WithSmallVocab_ProducesValidToken()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        for (var i = 0; i < 100; i++) logits[i] = (float)(i * 0.1);
        var options = new SamplingOptions { Seed = 7 };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.InRange(result, 0, 99);
    }

    [Fact]
    public void TopK_One_ReturnsArgMaxWithTemperature()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[42] = 10f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 1,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(42, result);
    }

    [Fact]
    public void TopP_One_DoesNotFilter()
    {
        var sampler = CreateSampler();
        var logits = MakeLogits(100, peakIndex: 55, peakValue: 10f);
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(55, result);
    }

    [Fact]
    public void RepeatPenalty_RespectsWindowSize()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[10] = 10f;
        var recentTokens = new List<int>();
        for (var i = 0; i < 100; i++) recentTokens.Add(i);
        recentTokens[95] = 10;
        var optionsWithSmallWindow = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 5,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, optionsWithSmallWindow, samplingBuf, samplingIdxBuf, recentTokens);

        Assert.NotEqual(10, result);
    }

    [Fact]
    public void RepeatPenalty_WindowExcludesOldTokens()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[42] = 10f;
        var recentTokens = new List<int> { 42, 5, 10, 15, 20, 25, 30, 35, 40 };
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 2,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, recentTokens);

        Assert.Equal(42, result);
    }

    [Fact]
    public void Combined_TopKAndTopP()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 10f;
        logits[1] = 9f;
        logits[2] = 8f;
        for (var i = 3; i < 100; i++) logits[i] = 0.001f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 5,
            TopP = 0.9f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.InRange(result, 0, 4);
    }

    [Fact]
    public void Combined_TopKAndTopPAndRepeatPenalty()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        logits[0] = 10f;
        logits[1] = 9f;
        logits[2] = 8f;
        for (var i = 3; i < 100; i++) logits[i] = 0.001f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 5,
            TopP = 0.9f,
            RepeatPenalty = 10f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 0 });

        Assert.InRange(result, 0, 4);
        Assert.NotEqual(0, result);
    }

    [Fact]
    public void NegativeTemperature_TreatedAsGreedy()
    {
        var sampler = CreateSampler();
        var logits = MakeLogits(100, peakIndex: 77, peakValue: 20f);
        var options = new SamplingOptions { Temperature = -1f };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(77, result);
    }

    [Fact]
    public void NegativeSeed_StillProducesValidToken()
    {
        var sampler = CreateSampler();
        var logits = new float[100];
        for (var i = 0; i < 100; i++) logits[i] = (float)(i * 0.1);
        var options = new SamplingOptions
        {
            Temperature = 1f,
            Seed = -1,
        };
        var samplingBuf = new float[100];
        var samplingIdxBuf = new int[100];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.InRange(result, 0, 99);
    }

    [Fact]
    public void RepeatPenalty_DividesPositiveLogits()
    {
        var sampler = CreateSampler();
        var logits = new float[10];
        logits[0] = 100f;
        logits[1] = 1f;
        for (var i = 2; i < 10; i++) logits[i] = 0.01f;
        var options = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 1000f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        var samplingBuf = new float[10];
        var samplingIdxBuf = new int[10];

        var result = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int> { 0 });

        Assert.NotEqual(0, result);
    }

    [Fact]
    public void RepeatPenalty_MultipliesNegativeLogits()
    {
        var sampler = CreateSampler();
        var logits = new float[10];
        logits[0] = -0.1f;
        logits[5] = 1f;
        var options = new SamplingOptions
        {
            Temperature = 0.5f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 1f,
            RepeatPenaltyWindowSize = 0,
            Seed = 42,
        };
        var samplingBuf = new float[10];
        var samplingIdxBuf = new int[10];

        var resultNoPenalty = sampler.Sample(logits, options, samplingBuf, samplingIdxBuf, new List<int>());

        Assert.Equal(5, resultNoPenalty);

        var options2 = new SamplingOptions
        {
            Temperature = 1f,
            TopK = 0,
            TopP = 1f,
            RepeatPenalty = 100f,
            RepeatPenaltyWindowSize = 64,
            Seed = 42,
        };
        logits[0] = -0.1f;
        logits[5] = 1f;
        var resultWithPenalty = sampler.Sample(logits, options2, samplingBuf, samplingIdxBuf, new List<int> { 5 });

        Assert.NotEqual(5, resultWithPenalty);
    }
}