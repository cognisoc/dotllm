using System.Runtime.InteropServices;
using Dotllm.Loading;
using Dotllm.Tensors;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Inference;

internal sealed class CpuBackend : IComputeBackend
{
    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols) =>
        TensorOps.MatMul(a, b, result, bType, aCols, bCols);

    public void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols) =>
        TensorOps.MatMulF32(a, b, result, aCols, bCols);

    public void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon) =>
        VectorMath.RmsNorm(input, weights, output, epsilon);

    public void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon) =>
        VectorMath.LayerNorm(input, weights, bias, output, epsilon);

    public void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim) =>
        TensorOps.ApplyRoPE(query, key, headDim, position, freqBase, rotaryDim);

    public void Softmax(Span<float> input, float? softcap = null) =>
        VectorMath.Softmax(input, softcap);

    public void Silu(ReadOnlySpan<float> input, Span<float> result) =>
        VectorMath.Silu(input, result);

    public void Gelu(ReadOnlySpan<float> input, Span<float> result) =>
        VectorMath.Gelu(input, result);
}