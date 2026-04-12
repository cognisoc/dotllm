using Dotllm.Loading;

namespace Dotllm.Tensors;

internal static class TensorSize
{
    public static ulong ByteCount(GgmlType type, ulong elementCount) => type switch
    {
        GgmlType.F32 => elementCount * 4,
        GgmlType.F16 => elementCount * 2,
        GgmlType.BF16 => elementCount * 2,
        GgmlType.Q4_0 => elementCount / 32 * 18,
        GgmlType.Q4_1 => elementCount / 32 * 20,
        GgmlType.Q5_0 => elementCount / 32 * 22,
        GgmlType.Q5_1 => elementCount / 32 * 24,
        GgmlType.Q8_0 => elementCount / 32 * 34,
        GgmlType.Q8_1 => elementCount / 32 * 40,
        GgmlType.Q2_K => elementCount / 256 * 84,
        GgmlType.Q3_K => elementCount / 256 * 110,
        GgmlType.Q4_K => elementCount / 256 * 144,
        GgmlType.Q5_K => elementCount / 256 * 176,
        GgmlType.Q6_K => elementCount / 256 * 210,
        _ => throw new NotSupportedException($"Unsupported GGML type: {type}"),
    };
}