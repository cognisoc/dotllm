using Llmdot.Inference;
using Xunit;

namespace Llmdot.Core.Tests.Inference;

public class ConvLayerTests
{
    [Fact]
    public void ConvStateCache_KernelSize1_NoHistory()
    {
        var cache = new ConvStateCache(1, 4, 1);
        // With kernelSize=1, historyLen=0, Store should be a no-op
        var data = new float[] { 1, 2, 3, 4 };
        cache.Store(0, data); // Should not crash
    }

    [Fact]
    public void ConvStateCache_KernelSize4_BuildsCorrectWindow()
    {
        var hiddenSize = 2;
        var kernelSize = 4;
        var cache = new ConvStateCache(1, hiddenSize, kernelSize);

        // Store 3 history items (kernelSize-1 = 3)
        cache.Store(0, new float[] { 1, 2 }); // pos 0
        cache.Store(0, new float[] { 3, 4 }); // pos 1
        cache.Store(0, new float[] { 5, 6 }); // pos 2

        // Build input for position 3
        var current = new float[] { 7, 8 };
        var input = cache.BuildInput(0, current, 3);

        // Should be kernelSize * hiddenSize = 8 elements
        Assert.Equal(kernelSize * hiddenSize, input.Length);

        // For channel 0: [history[0], history[1], history[2], current] = [1, 3, 5, 7]
        Assert.Equal(1f, input[0 * kernelSize + 0]); // ch0, pos0
        Assert.Equal(3f, input[0 * kernelSize + 1]); // ch0, pos1
        Assert.Equal(5f, input[0 * kernelSize + 2]); // ch0, pos2
        Assert.Equal(7f, input[0 * kernelSize + 3]); // ch0, current

        // For channel 1: [2, 4, 6, 8]
        Assert.Equal(2f, input[1 * kernelSize + 0]); // ch1, pos0
        Assert.Equal(4f, input[1 * kernelSize + 1]); // ch1, pos1
        Assert.Equal(6f, input[1 * kernelSize + 2]); // ch1, pos2
        Assert.Equal(8f, input[1 * kernelSize + 3]); // ch1, current
    }

    [Fact]
    public void ConvStateCache_EarlyPositions_ZeroPadded()
    {
        var hiddenSize = 2;
        var kernelSize = 3;
        var cache = new ConvStateCache(1, hiddenSize, kernelSize);

        // At position 0, there's no history
        var current = new float[] { 1, 2 };
        var input = cache.BuildInput(0, current, 0);

        // History positions are negative, should be zero-padded
        // ch0: [0, 0, 1]
        Assert.Equal(0f, input[0 * kernelSize + 0]); // ch0, pad
        Assert.Equal(0f, input[0 * kernelSize + 1]); // ch0, pad
        Assert.Equal(1f, input[0 * kernelSize + 2]); // ch0, current
    }

    [Fact]
    public void Conv1D_CpuBackend_IdentityKernel()
    {
        using var backend = new CpuBackend();
        // kernelSize=1, 2 channels, weight=[1, 1]
        var input = new float[] { 3.0f, 5.0f }; // [ch0_k0, ch1_k0]
        var weights = new float[] { 1.0f, 1.0f }; // [ch0_w0, ch1_w0]
        var output = new float[2];

        backend.Conv1D(input, weights, output, 1, 2);

        Assert.Equal(3.0f, output[0], 4);
        Assert.Equal(5.0f, output[1], 4);
    }

    [Fact]
    public void Conv1D_CpuBackend_AveragingKernel()
    {
        using var backend = new CpuBackend();
        var kernelSize = 3;
        var channels = 1;
        // input: 1 channel, kernel_size=3 -> [ch0_k0, ch0_k1, ch0_k2]
        var input = new float[] { 1.0f, 2.0f, 3.0f };
        var weights = new float[] { 1.0f / 3, 1.0f / 3, 1.0f / 3 };
        var output = new float[1];

        backend.Conv1D(input, weights, output, kernelSize, channels);

        Assert.Equal(2.0f, output[0], 2);
    }

    [Fact]
    public void Conv1D_CpuBackend_MultiChannel()
    {
        using var backend = new CpuBackend();
        var kernelSize = 2;
        var channels = 2;
        // input layout: [ch0_k0, ch0_k1, ch1_k0, ch1_k1]
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var weights = new float[] { 1.0f, 0.5f, 0.5f, 1.0f };
        var output = new float[2];

        backend.Conv1D(input, weights, output, kernelSize, channels);

        // ch0: 1*1 + 2*0.5 = 2.0
        // ch1: 3*0.5 + 4*1 = 5.5
        Assert.Equal(2.0f, output[0], 4);
        Assert.Equal(5.5f, output[1], 4);
    }
}
