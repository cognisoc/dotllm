using Dotllm.Inference;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Inference;

public class InferenceBuffersTests
{
    [Fact]
    public void Buffers_SmolLm2_135M_Dimensions()
    {
        var cfg = new TransformerConfig
        {
            HiddenSize = 576,
            HeadCount = 9,
            HeadCountKv = 3,
            HeadDim = 64,
            FfnDim = 1536,
            VocabSize = 49152,
            ContextLength = 2048,
            LayerCount = 30,
        };

        var buf = new InferenceBuffers(cfg);

        Assert.Equal(576, buf.HiddenState.Length);
        Assert.Equal(576, buf.NormBuf.Length);
        Assert.Equal(576, buf.QBuf.Length); // 9*64
        Assert.Equal(192, buf.KBuf.Length); // 3*64
        Assert.Equal(192, buf.VBuf.Length); // 3*64
        Assert.Equal(49152, buf.Logits.Length);
    }

    [Fact]
    public void Buffers_LargeVocab_LogitsBuffer()
    {
        var cfg = new TransformerConfig
        {
            HiddenSize = 2048,
            HeadCount = 32,
            HeadCountKv = 32,
            HeadDim = 64,
            FfnDim = 8192,
            VocabSize = 128000,
            ContextLength = 2048,
            LayerCount = 24,
        };

        var buf = new InferenceBuffers(cfg);
        Assert.Equal(128000, buf.Logits.Length);
    }

    [Fact]
    public void Buffers_SwiGLU_FfnBufSize()
    {
        var cfg = new TransformerConfig
        {
            HiddenSize = 576,
            HeadCount = 9,
            HeadCountKv = 3,
            HeadDim = 64,
            FfnDim = 1536,
            VocabSize = 1000,
            ContextLength = 128,
            LayerCount = 2,
        };

        var buf = new InferenceBuffers(cfg);
        // FfnBuf should be at least 2*FfnDim for SwiGLU gate+up
        Assert.True(buf.FfnBuf.Length >= 2 * cfg.FfnDim);
    }
}
