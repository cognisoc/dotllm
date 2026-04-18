using System;
using Llmdot.Loading;
using Llmdot.Tensors;
using Llmdot.Tensors.Dequantize;
using Llmdot.Tensors.Numeric;
using Xunit;

namespace Llmdot.Core.Tests.Tensors;

public class DequantizeTests
{
    [Fact]
    public void DequantizeToFloat_F32_IsIdentity()
    {
        var values = new float[] { 1.0f, -2.5f, 3.14f, 0.0f };
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        var dst = new float[values.Length];

        TensorOps.DequantizeToFloat(bytes, dst, GgmlType.F32, 1, values.Length);

        Assert.Equal(values, dst);
    }

    [Fact]
    public void DequantizeToFloat_Q4_0_ProducesCorrectOutputSize()
    {
        const int blockElements = 32;
        const int blockSizeBytes = 18;
        const int blocksPerRow = 2;
        const int rowElements = blockElements * blocksPerRow;
        const int numRows = 1;
        var src = new byte[numRows * blocksPerRow * blockSizeBytes];
        src[0] = 0x00; src[1] = 0x3C;
        src[18] = 0x00; src[19] = 0x3C;
        var dst = new float[numRows * rowElements];

        TensorOps.DequantizeToFloat(src, dst, GgmlType.Q4_0, numRows, rowElements);

        Assert.Equal(numRows * rowElements, dst.Length);
        Assert.All(dst, v => Assert.Equal(-8.0f, v, 0.001f));
    }

    [Fact]
    public void DequantizeToFloat_Q8_0_ProducesCorrectOutputSize()
    {
        const int blockElements = 32;
        const int blockSizeBytes = 34;
        const int blocksPerRow = 2;
        const int rowElements = blockElements * blocksPerRow;
        const int numRows = 1;
        var src = new byte[numRows * blocksPerRow * blockSizeBytes];
        src[0] = 0x00; src[1] = 0x3C;
        src[34] = 0x00; src[35] = 0x3C;
        var dst = new float[numRows * rowElements];

        TensorOps.DequantizeToFloat(src, dst, GgmlType.Q8_0, numRows, rowElements);

        Assert.Equal(numRows * rowElements, dst.Length);
        Assert.All(dst, v => Assert.Equal(0.0f, v, 0.001f));
    }

    [Fact]
    public void DequantizeF16_ConvertsKnownValues()
    {
        Assert.Equal(1.0f, HalfHelper.HalfToFloat((ushort)0x3C00));
        Assert.Equal(-1.0f, HalfHelper.HalfToFloat((ushort)0xBC00));
        Assert.Equal(0.5f, HalfHelper.HalfToFloat((ushort)0x3800));
        Assert.Equal(0.0f, HalfHelper.HalfToFloat((ushort)0x0000));
    }

    [Fact]
    public void DequantizeBF16_ConvertsKnownValues()
    {
        var src1 = new byte[] { 0x80, 0x3F };
        var dst1 = new float[1];
        DequantizeBF16.Dequantize(src1, dst1, 1);
        Assert.Equal(1.0f, dst1[0], 0.001f);

        var src2 = new byte[] { 0x00, 0x40 };
        var dst2 = new float[1];
        DequantizeBF16.Dequantize(src2, dst2, 1);
        Assert.Equal(2.0f, dst2[0], 0.001f);

        var srcNeg1 = new byte[] { 0x80, 0xBF };
        var dstNeg1 = new float[1];
        DequantizeBF16.Dequantize(srcNeg1, dstNeg1, 1);
        Assert.Equal(-1.0f, dstNeg1[0], 0.001f);
    }
}