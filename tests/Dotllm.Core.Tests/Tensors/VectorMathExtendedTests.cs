using Dotllm.Tensors.Numeric;
using Xunit;

namespace Dotllm.Core.Tests.Tensors;

public class VectorMathExtendedTests
{
    [Fact]
    public void SiluInPlace_MatchesSiluOutOfPlace()
    {
        var input = new float[] { -3f, -1f, 0f, 0.5f, 1f, 2f, 5f, 10f };
        var expected = new float[input.Length];
        VectorMath.Silu(input, expected);
        var actual = new Span<float>(new float[input.Length]);
        input.CopyTo(actual);
        VectorMath.SiluInPlace(actual);
        for (var i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 0.0001f);
    }

    [Fact]
    public void Gelu_Vector_MatchesScalarGelu()
    {
        var input = new float[] { -1f, 0f, 0.5f, 1f };
        var result = new float[input.Length];
        VectorMath.Gelu(input, result);
        for (var i = 0; i < input.Length; i++)
        {
            var expected = VectorMath.Gelu(input[i]);
            Assert.Equal(expected, result[i], 0.05f);
        }
    }

    [Fact]
    public void Gelu_Scalar_Zero_ReturnsZero()
    {
        Assert.Equal(0f, VectorMath.Gelu(0f), 0.0001f);
    }

    [Fact]
    public void Gelu_Scalar_PositiveValue_ReturnsPositive()
    {
        var result = VectorMath.Gelu(1f);
        Assert.True(result > 0f);
        Assert.True(result < 1f);
    }

    [Fact]
    public void Gelu_Scalar_NegativeValue_ReturnsNegative()
    {
        var result = VectorMath.Gelu(-1f);
        Assert.True(result < 0f);
        Assert.True(result > -1f);
    }

    [Fact]
    public void Mul_Elementwise_WritesProduct()
    {
        var a = new float[] { 1f, 2f, 3f, 4f };
        var b = new float[] { 5f, 6f, 7f, 8f };
        var result = new float[4];
        VectorMath.Mul(a, b, result);
        Assert.Equal(new float[] { 5f, 12f, 21f, 32f }, result);
    }

    [Fact]
    public void Scale_InPlace_ModifiesSpan()
    {
        var input = new float[] { 1f, 2f, 3f, 4f, 5f };
        VectorMath.Scale(input, 3f);
        Assert.Equal(new float[] { 3f, 6f, 9f, 12f, 15f }, input);
    }

    [Fact]
    public void Add_InPlace_ModifiesFirstSpan()
    {
        var a = new float[] { 1f, 2f, 3f, 4f };
        var b = new float[] { 10f, 20f, 30f, 40f };
        VectorMath.Add(a, b);
        Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, a);
    }

    [Fact]
    public void Softcap_Standalone_MatchesCapTimesTanh()
    {
        var values = new float[] { -2f, 0f, 1f, 2f };
        var cap = 3f;
        var input = new Span<float>(new float[values.Length]);
        values.CopyTo(input);
        VectorMath.Softcap(input, cap);
        for (var i = 0; i < values.Length; i++)
        {
            var expected = cap * MathF.Tanh(values[i] / cap);
            Assert.Equal(expected, input[i], 0.05f);
        }
    }
}