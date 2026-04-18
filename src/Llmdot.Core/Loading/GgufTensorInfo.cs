namespace Llmdot.Loading;

internal sealed class GgufTensorInfo
{
    public string Name { get; init; } = string.Empty;
    public uint[] Dimensions { get; init; } = [];
    public GgmlType Type { get; init; }
    public ulong Offset { get; init; }
    public ulong ElementCount { get; init; }
}