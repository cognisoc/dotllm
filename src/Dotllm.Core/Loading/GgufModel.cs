namespace Dotllm.Loading;

internal sealed class GgufModel
{
    public uint Version { get; init; }
    public ulong TensorCount { get; init; }
    public ulong Metadata_kvCount { get; init; }
    public GgufMetadata Metadata { get; init; } = null!;
    public IReadOnlyList<GgufTensorInfo> TensorInfos { get; init; } = [];
    public ulong TensorDataOffset { get; init; }
}