using Llmdot.Tensors;
using Xunit;

namespace Llmdot.Core.Tests.Tensors;

public class TensorOpsTests
{
    [Fact]
    public void MatMulF32_IdentityMatrix_ReturnsInput()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f };
        var result = new float[3];

        TensorOps.MatMulF32(a, b, result, aCols: 3, bCols: 3);

        Assert.Equal(new float[] { 1f, 2f, 3f }, result);
    }

    [Fact]
    public void MatMulF32_2x3Times3x2()
    {
        var a = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var b = new float[] { 7f, 8f, 9f, 10f, 11f, 12f };
        var result = new float[4];

        TensorOps.MatMulF32(a, b, result, aCols: 3, bCols: 2);

        Assert.Equal(50f, result[0], 0.001f);
        Assert.Equal(68f, result[1], 0.001f);
        Assert.Equal(122f, result[2], 0.001f);
        Assert.Equal(167f, result[3], 0.001f);
    }

    [Fact]
    public void ApplyRoPE_ModifiesQueryAndKey()
    {
        var q = new float[128];
        var k = new float[128];
        for (var i = 0; i < 128; i++) { q[i] = 1f; k[i] = 1f; }

        TensorOps.ApplyRoPE(q, k, headDim: 128, position: 1, freqBase: 10000f, rotaryDim: 128);

        Assert.NotEqual(1f, q[64]);
        Assert.NotEqual(1f, k[64]);
    }

    [Fact]
    public void ApplyRoPE_PositionZero_CosineOneForLowFreq()
    {
        var q = new float[4];
        var k = new float[4];
        q[0] = 1f; q[1] = 0f; q[2] = 1f; q[3] = 0f;
        k[0] = 1f; k[1] = 0f; k[2] = 1f; k[3] = 0f;

        TensorOps.ApplyRoPE(q, k, headDim: 4, position: 0, freqBase: 10000f, rotaryDim: 4);

        Assert.Equal(1f, q[0], 0.001f);
        Assert.Equal(0f, q[1], 0.001f);
        Assert.Equal(1f, k[0], 0.001f);
    }

    [Fact]
    public void ApplyRoPE_SameSpan_AppliesOnce()
    {
        var data = new float[128];
        for (var i = 0; i < 128; i++) data[i] = 1f;
        var separate = new float[128];
        Array.Copy(data, separate, 128);

        var span = data.AsSpan();
        TensorOps.ApplyRoPE(span, span, headDim: 128, position: 1, freqBase: 10000f, rotaryDim: 128);

        var q = new float[128];
        var k = new float[128];
        for (var i = 0; i < 128; i++) { q[i] = 1f; k[i] = 1f; }
        TensorOps.ApplyRoPE(q, k, headDim: 128, position: 1, freqBase: 10000f, rotaryDim: 128);

        for (var i = 0; i < 128; i++)
            Assert.Equal(q[i], data[i], 0.0001f);
    }
}