using System.Buffers.Binary;
using System.Text;

namespace Dotllm.Loading;

internal sealed class GgufReader
{
    public static GgufModel Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var magicBytes = reader.ReadBytes(4);
        if (magicBytes.Length < 4 || magicBytes[0] != 'G' || magicBytes[1] != 'G' || magicBytes[2] != 'U' || magicBytes[3] != 'F')
            throw new InvalidDataException("Invalid GGUF magic. Expected 'GGUF'.");

        var version = reader.ReadUInt32LittleEndian();
        if (version < GgufConstants.MinimumVersion)
            throw new InvalidDataException($"Unsupported GGUF version: {version}. Minimum supported: {GgufConstants.MinimumVersion}.");

        var tensorCount = reader.ReadUInt64LittleEndian();
        var metadataKvCount = reader.ReadUInt64LittleEndian();

        var metadata = ReadMetadata(reader, metadataKvCount);
        var tensorInfos = ReadTensorInfos(reader, tensorCount);

        var alignment = (uint)metadata.GetOrDefault("general.alignment", 32u);
        var tensorDataOffset = (ulong)stream.Position;
        var padding = (alignment - (tensorDataOffset % alignment)) % alignment;
        tensorDataOffset += padding;

        return new GgufModel
        {
            Version = version,
            TensorCount = tensorCount,
            Metadata_kvCount = metadataKvCount,
            Metadata = metadata,
            TensorInfos = tensorInfos,
            TensorDataOffset = tensorDataOffset,
        };
    }

    private static GgufMetadata ReadMetadata(BinaryReader reader, ulong count)
    {
        var values = new Dictionary<string, GgufMetadataValue>((int)count);

        for (ulong i = 0; i < count; i++)
        {
            var key = ReadGgufString(reader);
            var valueType = (GgufValueType)reader.ReadUInt32LittleEndian();
            var value = ReadMetadataValue(reader, valueType);
            values[key] = new GgufMetadataValue { Type = valueType, Value = value };
        }

        return new GgufMetadata(values);
    }

    private static object ReadMetadataValue(BinaryReader reader, GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => reader.ReadByte(),
        GgufValueType.Int8 => reader.ReadSByte(),
        GgufValueType.UInt16 => reader.ReadUInt16LittleEndian(),
        GgufValueType.Int16 => reader.ReadInt16LittleEndian(),
        GgufValueType.UInt32 => reader.ReadUInt32LittleEndian(),
        GgufValueType.Int32 => reader.ReadInt32LittleEndian(),
        GgufValueType.Float32 => reader.ReadSingleLittleEndian(),
        GgufValueType.Bool => reader.ReadUInt32LittleEndian() != 0,
        GgufValueType.String => ReadGgufString(reader),
        GgufValueType.UInt64 => reader.ReadUInt64LittleEndian(),
        GgufValueType.Int64 => reader.ReadInt64LittleEndian(),
        GgufValueType.Float64 => reader.ReadDoubleLittleEndian(),
        GgufValueType.Array => ReadGgufArray(reader),
        _ => throw new InvalidDataException($"Unsupported GGUF value type: {type}"),
    };

    private static string ReadGgufString(BinaryReader reader)
    {
        var length = reader.ReadUInt64LittleEndian();
        var bytes = reader.ReadBytes((int)length);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object[] ReadGgufArray(BinaryReader reader)
    {
        var elementType = (GgufValueType)reader.ReadUInt32LittleEndian();
        var count = reader.ReadUInt64LittleEndian();
        var array = new object[(int)count];

        for (var i = 0; i < (int)count; i++)
        {
            array[i] = ReadMetadataValue(reader, elementType);
        }

        return array;
    }

    private static List<GgufTensorInfo> ReadTensorInfos(BinaryReader reader, ulong count)
    {
        var infos = new List<GgufTensorInfo>((int)count);

        for (ulong i = 0; i < count; i++)
        {
            var name = ReadGgufString(reader);
            var nDimensions = reader.ReadUInt32LittleEndian();
            var dimensions = new uint[nDimensions];
            for (var d = 0; d < nDimensions; d++)
                dimensions[d] = reader.ReadUInt32LittleEndian();

            var type = (GgmlType)reader.ReadUInt32LittleEndian();
            var offset = reader.ReadUInt64LittleEndian();

            var elementCount = dimensions.Length > 0
                ? dimensions.Aggregate(1u, (acc, dim) => acc * dim)
                : 0;

            infos.Add(new GgufTensorInfo
            {
                Name = name,
                Dimensions = dimensions,
                Type = type,
                Offset = offset,
                ElementCount = elementCount,
            });
        }

        return infos;
    }
}

internal static class BinaryReaderExtensions
{
    public static ushort ReadUInt16LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadUInt16LittleEndian(reader.ReadBytes(2));

    public static short ReadInt16LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadInt16LittleEndian(reader.ReadBytes(2));

    public static uint ReadUInt32LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadUInt32LittleEndian(reader.ReadBytes(4));

    public static int ReadInt32LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadInt32LittleEndian(reader.ReadBytes(4));

    public static ulong ReadUInt64LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadUInt64LittleEndian(reader.ReadBytes(8));

    public static long ReadInt64LittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadInt64LittleEndian(reader.ReadBytes(8));

    public static float ReadSingleLittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadSingleLittleEndian(reader.ReadBytes(4));

    public static double ReadDoubleLittleEndian(this BinaryReader reader) =>
        BinaryPrimitives.ReadDoubleLittleEndian(reader.ReadBytes(8));
}