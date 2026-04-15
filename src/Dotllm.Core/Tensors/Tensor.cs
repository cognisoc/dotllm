using Dotllm.Loading;

namespace Dotllm.Tensors;

internal sealed class Tensor
{
    public string Name { get; init; } = string.Empty;
    public uint[] Dimensions { get; init; } = [];
    public GgmlType ElementType { get; init; }
    public Memory<byte> Data { get; init; }
    public ulong ElementCount { get; init; }
    public int RowCount => Dimensions.Length > 1 ? (int)Dimensions[1] : 1;
    public int ColumnCount => Dimensions.Length > 0 ? (int)Dimensions[0] : 0;
}