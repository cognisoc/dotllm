using Dotllm.Loading;
using Dotllm.Tensors;
using Xunit;

namespace Dotllm.Core.Tests.Tensors;

public class TensorSizeTests
{
    [Theory]
    [InlineData((uint)GgmlType.F32, 1024UL, 4096UL)]
    [InlineData((uint)GgmlType.F16, 1024UL, 2048UL)]
    [InlineData((uint)GgmlType.Q8_0, 256UL, 272UL)]
    [InlineData((uint)GgmlType.Q4_0, 256UL, 144UL)]
    public void ByteCount_ReturnsExpectedSize(uint typeValue, ulong elements, ulong expectedBytes)
    {
        var type = (GgmlType)typeValue;
        var actual = TensorSize.ByteCount(type, elements);
        Assert.Equal(expectedBytes, actual);
    }

    [Fact]
    public void ByteCount_UnsupportedType_ThrowsNotSupportedException()
    {
        Assert.Throws<NotSupportedException>(() =>
            TensorSize.ByteCount((GgmlType)999, 1024));
    }
}