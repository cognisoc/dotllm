namespace Dotllm.Inference;

internal interface IComputeBackend
{
    void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, Loading.GgmlType bType, int aCols, int bCols);
    void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols);
    void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon);
    void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon);
    void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim);
    void Softmax(Span<float> input, float? softcap = null);
    void Silu(ReadOnlySpan<float> input, Span<float> result);
    void Gelu(ReadOnlySpan<float> input, Span<float> result);
}