using Dotllm.Inference;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Inference;

public class InferenceEngineUnitTests
{
    [Fact]
    public void KvCache_CorrectDimensions_ForSmolLm2()
    {
        var cfg = new TransformerConfig
        {
            HiddenSize = 576,
            HeadCount = 9,
            HeadCountKv = 3,
            HeadDim = 64,
            LayerCount = 30,
            ContextLength = 2048,
        };

        var kvDim = cfg.KvDim; // HeadCountKv * HeadDim = 3*64 = 192
        Assert.Equal(192, kvDim);

        var cache = new KvCache(cfg.LayerCount, kvDim, cfg.ContextLength);
        Assert.Equal(0, cache.CurrentPosition);
    }

    [Fact]
    public void KvCache_CorrectDimensions_ForLfm2()
    {
        var cfg = new TransformerConfig
        {
            HiddenSize = 2048,
            HeadCount = 16,
            HeadCountKv = 8,
            HeadDim = 128,
            LayerCount = 16,
            ContextLength = 32768,
        };

        var kvDim = cfg.KvDim; // 8*128 = 1024
        Assert.Equal(1024, kvDim);
    }

    [Fact]
    public void KvCache_Advance_IncrementsPosition()
    {
        var cache = new KvCache(2, 64, 128);
        Assert.Equal(0, cache.CurrentPosition);
        cache.Advance();
        Assert.Equal(1, cache.CurrentPosition);
        cache.Advance();
        Assert.Equal(2, cache.CurrentPosition);
    }

    [Fact]
    public void KvCache_Reset_ZerosPosition()
    {
        var cache = new KvCache(2, 64, 128);
        cache.Advance();
        cache.Advance();
        Assert.Equal(2, cache.CurrentPosition);
        cache.Reset();
        Assert.Equal(0, cache.CurrentPosition);
    }
}
