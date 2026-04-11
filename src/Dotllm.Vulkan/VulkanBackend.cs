using System.Runtime.InteropServices;
using Dotllm.Loading;
using Dotllm.Tensors.Numeric;
using Dotllm.Inference;

namespace Dotllm.Vulkan;

public sealed class VulkanBackend : IComputeBackend
{
    private readonly VulkanBackendOptions _options;
    private bool _disposed;

    public VulkanBackend(VulkanBackendOptions? options = null)
    {
        _options = options ?? new VulkanBackendOptions();
    }

    public bool IsAvailable => VulkanRuntime.IsAvailable;

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
    {
        FallbackMatMul(a, b, result, bType, aCols, bCols);
    }

    public void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        FallbackMatMulF32(a, b, result, aCols, bCols);
    }

    public void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon) =>
        VectorMath.RmsNorm(input, weights, output, epsilon);

    public void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon) =>
        VectorMath.LayerNorm(input, weights, bias, output, epsilon);

    public void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim) =>
        Tensors.TensorOps.ApplyRoPE(query, key, headDim, position, freqBase, rotaryDim);

    public void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, ReadOnlySpan<float> freqTable) =>
        Tensors.TensorOps.ApplyRoPE(query, key, headDim, position, freqTable);

    public void Softmax(Span<float> input, float? softcap = null) =>
        VectorMath.Softmax(input, softcap);

    public void Silu(ReadOnlySpan<float> input, Span<float> result) =>
        VectorMath.Silu(input, result);

    public void SiluInPlace(Span<float> input) =>
        VectorMath.SiluInPlace(input);

    public void Gelu(ReadOnlySpan<float> input, Span<float> result) =>
        VectorMath.Gelu(input, result);

    public float GeluScalar(float x) =>
        VectorMath.Gelu(x);

    public void Add(Span<float> a, ReadOnlySpan<float> b) =>
        VectorMath.Add(a, b);

    public void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) =>
        VectorMath.Add(a, b, result);

    public void Scale(ReadOnlySpan<float> input, float scale, Span<float> result) =>
        VectorMath.Scale(input, scale, result);

    public void Scale(Span<float> input, float scale) =>
        VectorMath.Scale(input, scale);

    public void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) =>
        VectorMath.Mul(a, b, result);

    public void Softcap(Span<float> input, float cap) =>
        VectorMath.Softcap(input, cap);

    public void Conv1D(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int kernelSize, int inputDim) =>
        Tensors.TensorOps.Conv1D(input, weights, output, kernelSize, inputDim);

    public void DequantizeToFloat(ReadOnlySpan<byte> src, Span<float> dst, GgmlType type, int numRows, int rowElements) =>
        Tensors.TensorOps.DequantizeToFloat(src, dst, type, numRows, rowElements);

    public int ArgMax(ReadOnlySpan<float> input) =>
        VectorMath.ArgMax(input);

    private static void FallbackMatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols) =>
        Tensors.TensorOps.MatMul(a, b, result, bType, aCols, bCols);

    private static void FallbackMatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols) =>
        Tensors.TensorOps.MatMulF32(a, b, result, aCols, bCols);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}