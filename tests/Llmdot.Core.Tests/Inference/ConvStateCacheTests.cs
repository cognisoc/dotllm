using Llmdot.Inference;
using Xunit;

namespace Llmdot.Core.Tests.Inference;

public class ConvStateCacheTests
{
    [Fact]
    public void Store_And_BuildInput_Returns_Stored_Data()
    {
        var cache = new ConvStateCache(layerCount: 1, hiddenSize: 2, kernelSize: 3);
        float[] data0 = [1f, 2f];
        float[] data1 = [3f, 4f];
        float[] current = [5f, 6f];

        cache.Store(0, data0);
        cache.Store(0, data1);

        var result = cache.BuildInput(0, current, position: 2);

        Assert.Equal(1f, result[0]);
        Assert.Equal(3f, result[1]);
        Assert.Equal(5f, result[2]);
        Assert.Equal(2f, result[3]);
        Assert.Equal(4f, result[4]);
        Assert.Equal(6f, result[5]);
    }

    [Fact]
    public void Rolling_Buffer_Overwrites_Oldest_Entry()
    {
        var cache = new ConvStateCache(layerCount: 1, hiddenSize: 1, kernelSize: 3);
        float[] oldest = [1f];
        float[] middle = [2f];
        float[] newest = [3f];
        float[] current = [4f];

        cache.Store(0, oldest);
        cache.Store(0, middle);
        cache.Store(0, newest);

        var result = cache.BuildInput(0, current, position: 3);

        Assert.Equal(2f, result[0]);
        Assert.Equal(3f, result[1]);
        Assert.Equal(4f, result[2]);
    }

    [Fact]
    public void Reset_Clears_State()
    {
        var cache = new ConvStateCache(layerCount: 1, hiddenSize: 1, kernelSize: 3);
        float[] data = [42f];
        float[] current = [99f];

        cache.Store(0, data);
        cache.Reset();

        var result = cache.BuildInput(0, current, position: 0);

        Assert.Equal(0f, result[0]);
        Assert.Equal(0f, result[1]);
        Assert.Equal(99f, result[2]);
    }

    [Fact]
    public void Multi_Layer_Isolation()
    {
        var cache = new ConvStateCache(layerCount: 2, hiddenSize: 1, kernelSize: 3);
        float[] layer0Data0 = [10f];
        float[] layer0Data1 = [20f];
        float[] current = [50f];

        cache.Store(0, layer0Data0);
        cache.Store(0, layer0Data1);

        var layer0Result = cache.BuildInput(0, current, position: 2);
        Assert.Equal(10f, layer0Result[0]);
        Assert.Equal(20f, layer0Result[1]);
        Assert.Equal(50f, layer0Result[2]);

        var layer1Result = cache.BuildInput(1, current, position: 2);
        Assert.Equal(0f, layer1Result[0]);
        Assert.Equal(0f, layer1Result[1]);
        Assert.Equal(50f, layer1Result[2]);
    }
}