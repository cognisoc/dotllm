namespace Dotllm.Inference;

internal interface IComputeBackend
{
    void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, Loading.GgmlType bType, int aCols, int bCols);
    void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols);
    void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon);
    void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon);
    void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim);
    void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, ReadOnlySpan<float> freqTable);
    void Softmax(Span<float> input, float? softcap = null);
    void Silu(ReadOnlySpan<float> input, Span<float> result);
    void SiluInPlace(Span<float> input);
    void Gelu(ReadOnlySpan<float> input, Span<float> result);
    float GeluScalar(float x);
    void Add(Span<float> a, ReadOnlySpan<float> b);
    void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result);
    void Scale(ReadOnlySpan<float> input, float scale, Span<float> result);
    void Scale(Span<float> input, float scale);
    void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result);
    void Softcap(Span<float> input, float cap);
    void Conv1D(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int kernelSize, int inputDim);
    void DequantizeToFloat(ReadOnlySpan<byte> src, Span<float> dst, Loading.GgmlType type, int numRows, int rowElements);
    int ArgMax(ReadOnlySpan<float> input);
}