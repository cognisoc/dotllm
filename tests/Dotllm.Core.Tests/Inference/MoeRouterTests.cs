using Dotllm.Inference;
using Dotllm.Loading;
using System.Runtime.InteropServices;
using Xunit;

namespace Dotllm.Core.Tests.Inference;

public class MoeRouterTests
{
    [Fact]
    public void Route_SelectsTopK_FromGateLogits()
    {
        using var backend = new CpuBackend();
        var router = new MoeRouter(backend);

        var hiddenSize = 4;
        var expertCount = 4;
        var topK = 2;

        // Create identity-like gate weights so gateLogits ≈ input values mapped to experts
        // gate weight is expertCount x hiddenSize stored row-major as F32
        var gateWeights = new float[expertCount * hiddenSize];
        // Expert 0: responds to dim 0, Expert 1: dim 1, etc.
        for (var i = 0; i < expertCount; i++)
            gateWeights[i * hiddenSize + i] = 1.0f;

        var gateBytes = MemoryMarshal.AsBytes(gateWeights.AsSpan()).ToArray();
        var hiddenState = new float[] { 0.1f, 0.9f, 0.5f, 0.2f };

        var gateLogits = new float[expertCount];
        var selectedExperts = new int[topK];
        var routingWeights = new float[topK];

        router.Route(hiddenState, gateBytes, GgmlType.F32, hiddenSize, expertCount, topK,
            gateLogits, selectedExperts, routingWeights);

        // Expert 1 (logit=0.9) and Expert 2 (logit=0.5) should be selected
        Assert.Equal(1, selectedExperts[0]);
        Assert.Equal(2, selectedExperts[1]);

        // Weights should sum to 1.0
        var sum = routingWeights[0] + routingWeights[1];
        Assert.InRange(sum, 0.99f, 1.01f);

        // First weight should be larger (higher logit)
        Assert.True(routingWeights[0] > routingWeights[1]);
    }

    [Fact]
    public void Route_SingleExpert_ReturnsFullWeight()
    {
        using var backend = new CpuBackend();
        var router = new MoeRouter(backend);

        var hiddenSize = 2;
        var expertCount = 3;
        var topK = 1;

        var gateWeights = new float[expertCount * hiddenSize];
        gateWeights[0 * hiddenSize + 0] = 1.0f; // Expert 0 responds to dim 0
        gateWeights[1 * hiddenSize + 1] = 10.0f; // Expert 1 responds strongly to dim 1
        gateWeights[2 * hiddenSize + 0] = 0.5f; // Expert 2 responds weakly to dim 0

        var gateBytes = MemoryMarshal.AsBytes(gateWeights.AsSpan()).ToArray();
        var hiddenState = new float[] { 1.0f, 1.0f };

        var gateLogits = new float[expertCount];
        var selectedExperts = new int[topK];
        var routingWeights = new float[topK];

        router.Route(hiddenState, gateBytes, GgmlType.F32, hiddenSize, expertCount, topK,
            gateLogits, selectedExperts, routingWeights);

        Assert.Equal(1, selectedExperts[0]); // Expert 1 has highest gate value
        Assert.Equal(1.0f, routingWeights[0]); // Single expert gets full weight
    }

    [Fact]
    public void Route_EqualGateLogits_SelectsFirstK()
    {
        using var backend = new CpuBackend();
        var router = new MoeRouter(backend);

        var hiddenSize = 2;
        var expertCount = 4;
        var topK = 2;

        // All experts have equal gate weights
        var gateWeights = new float[expertCount * hiddenSize];
        for (var i = 0; i < expertCount; i++)
        {
            gateWeights[i * hiddenSize + 0] = 1.0f;
            gateWeights[i * hiddenSize + 1] = 1.0f;
        }

        var gateBytes = MemoryMarshal.AsBytes(gateWeights.AsSpan()).ToArray();
        var hiddenState = new float[] { 1.0f, 1.0f };

        var gateLogits = new float[expertCount];
        var selectedExperts = new int[topK];
        var routingWeights = new float[topK];

        router.Route(hiddenState, gateBytes, GgmlType.F32, hiddenSize, expertCount, topK,
            gateLogits, selectedExperts, routingWeights);

        // With equal logits, should select first K experts (0, 1)
        Assert.Equal(0, selectedExperts[0]);
        Assert.Equal(1, selectedExperts[1]);

        // Equal weights
        Assert.InRange(routingWeights[0], 0.49f, 0.51f);
        Assert.InRange(routingWeights[1], 0.49f, 0.51f);
    }

    [Fact]
    public void MoeBuffers_AllocatesCorrectSizes()
    {
        var cfg = new Dotllm.Models.TransformerConfig
        {
            HiddenSize = 256,
            HeadCount = 4,
            HeadCountKv = 4,
            HeadDim = 64,
            FfnDim = 512,
            VocabSize = 1000,
            ContextLength = 128,
            LayerCount = 2,
            ExpertCount = 8,
            ExpertUsedCount = 2,
        };

        var buffers = new InferenceBuffers(cfg);

        Assert.Equal(8, buffers.MoeGateLogits.Length);
        Assert.Equal(256, buffers.MoeExpertResultBuf.Length);
        Assert.Equal(256, buffers.MoeAccumulatorBuf.Length);
        Assert.Equal(2, buffers.MoeSelectedExperts.Length);
        Assert.Equal(2, buffers.MoeRoutingWeights.Length);
    }
}
