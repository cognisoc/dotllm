using Llmdot.Tensors.Numeric;
using Xunit;

namespace Llmdot.Core.Tests.Tensors;

public class VectorMathTests
{
    [Fact]
    public void RmsNorm_UnitWeights_PreservesNormalizedInput()
    {
        var input = new float[] { 3f, 4f };
        var weights = new float[] { 1f, 1f };
        var output = new float[2];

        VectorMath.RmsNorm(input, weights, output, 1e-5f);

        var expectedMag = MathF.Sqrt((9f + 16f) / 2f + 1e-5f);
        Assert.Equal(3f / expectedMag, output[0], 0.001f);
        Assert.Equal(4f / expectedMag, output[1], 0.001f);
    }

    [Fact]
    public void LayerNorm_ZeroBias_OutputIsNormalized()
    {
        var input = new float[] { 1f, 2f, 3f };
        var weights = new float[] { 1f, 1f, 1f };
        var bias = new float[] { 0f, 0f, 0f };
        var output = new float[3];

        VectorMath.LayerNorm(input, weights, bias, output, 1e-5f);

        var mean = output.Average();
        Assert.InRange(mean, -0.01f, 0.01f);
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        var input = new float[] { 1f, 2f, 3f, 4f };

        VectorMath.Softmax(input);

        var sum = input.Sum();
        Assert.Equal(1f, sum, 0.001f);
    }

    [Fact]
    public void Softmax_WithSoftcap_StillSumsToOne()
    {
        var input = new float[] { 10f, 20f, 30f, 40f };

        VectorMath.Softmax(input, softcap: 5f);

        var sum = input.Sum();
        Assert.Equal(1f, sum, 0.001f);
    }

    [Fact]
    public void Silu_Zero_ReturnsZero()
    {
        var input = new float[] { 0f };
        var output = new float[1];

        VectorMath.Silu(input, output);

        Assert.Equal(0f, output[0], 0.001f);
    }

    [Fact]
    public void Silu_Positive_ApproachesInput()
    {
        var input = new float[] { 100f };
        var output = new float[1];

        VectorMath.Silu(input, output);

        Assert.Equal(100f, output[0], 0.1f);
    }

    [Fact]
    public void ArgMax_ReturnsIndex()
    {
        var input = new float[] { 1f, 5f, 3f, 2f };
        var result = VectorMath.ArgMax(input);
        Assert.Equal(1, result);
    }

    [Fact]
    public void Add_Elementwise()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 4f, 5f, 6f };
        var result = new float[3];

        VectorMath.Add(a, b, result);

        Assert.Equal(new float[] { 5f, 7f, 9f }, result);
    }

    [Fact]
    public void Scale_MultipliesByScalar()
    {
        var input = new float[] { 1f, 2f, 3f };
        var result = new float[3];

        VectorMath.Scale(input, 2f, result);

        Assert.Equal(new float[] { 2f, 4f, 6f }, result);
    }
}