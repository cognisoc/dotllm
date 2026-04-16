using Dotllm.Inference;
using Xunit;

namespace Dotllm.Core.Tests.Inference;

public class KvCacheTests
{
    private const int LayerCount = 2;
    private const int KvDim = 4;
    private const int MaxSeqLen = 8;

    [Fact]
    public void GetKeySlot_ReturnsSpanOfKvDimLength()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        var slot = cache.GetKeySlot(0, 0);
        Assert.Equal(KvDim, slot.Length);
    }

    [Fact]
    public void GetValueSlot_ReturnsSpanOfKvDimLength()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        var slot = cache.GetValueSlot(0, 0);
        Assert.Equal(KvDim, slot.Length);
    }

    [Fact]
    public void GetKeys_ReturnsSpanOfSeqLenTimesKvDim()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        cache.Advance(3);
        var keys = cache.GetKeys(0, 3);
        Assert.Equal(3 * KvDim, keys.Length);
    }

    [Fact]
    public void GetValues_ReturnsSpanOfSeqLenTimesKvDim()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        cache.Advance(3);
        var values = cache.GetValues(0, 3);
        Assert.Equal(3 * KvDim, values.Length);
    }

    [Fact]
    public void Advance_IncrementsCurrentPosition()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        Assert.Equal(0, cache.CurrentPosition);
        cache.Advance();
        Assert.Equal(1, cache.CurrentPosition);
        cache.Advance(3);
        Assert.Equal(4, cache.CurrentPosition);
    }

    [Fact]
    public void Reset_SetsCurrentPositionToZero()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        cache.Advance(5);
        Assert.Equal(5, cache.CurrentPosition);
        cache.Reset();
        Assert.Equal(0, cache.CurrentPosition);
    }

    [Fact]
    public void WrittenData_CanBeReadBackThroughGetKeys()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var slot = cache.GetKeySlot(0, 0);
        data.CopyTo(slot);
        cache.Advance();

        var keys = cache.GetKeys(0, 1);
        Assert.Equal(1.0f, keys[0]);
        Assert.Equal(2.0f, keys[1]);
        Assert.Equal(3.0f, keys[2]);
        Assert.Equal(4.0f, keys[3]);
    }

    [Fact]
    public void WrittenData_CanBeReadBackThroughGetValues()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);
        var data = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
        var slot = cache.GetValueSlot(0, 0);
        data.CopyTo(slot);
        cache.Advance();

        var values = cache.GetValues(0, 1);
        Assert.Equal(5.0f, values[0]);
        Assert.Equal(6.0f, values[1]);
        Assert.Equal(7.0f, values[2]);
        Assert.Equal(8.0f, values[3]);
    }

    [Fact]
    public void MultiLayer_DataInLayer0DoesNotAffectLayer1()
    {
        var cache = new KvCache(LayerCount, KvDim, MaxSeqLen);

        var keySlot0 = cache.GetKeySlot(0, 0);
        keySlot0[0] = 42.0f;

        var keySlot1 = cache.GetKeySlot(1, 0);
        keySlot1[0] = 99.0f;

        cache.Advance();

        var keys0 = cache.GetKeys(0, 1);
        var keys1 = cache.GetKeys(1, 1);
        Assert.Equal(42.0f, keys0[0]);
        Assert.Equal(99.0f, keys1[0]);
    }
}