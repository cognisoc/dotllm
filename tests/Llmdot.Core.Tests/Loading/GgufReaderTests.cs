using Llmdot.Loading;
using Xunit;

namespace Llmdot.Core.Tests.Loading;

public class GgufReaderTests
{
    [Fact]
    public void Read_InvalidMagic_ThrowsInvalidDataException()
    {
        using var ms = new MemoryStream();
        var writer = new BinaryWriter(ms);
        writer.Write(new byte[] { 0x00, 0x00, 0x00, 0x00 });
        writer.Flush();
        ms.Position = 0;

        Assert.Throws<InvalidDataException>(() => Llmdot.Loading.GgufReader.Read(ms));
    }

    [Fact]
    public void Read_ValidMagic_ParsesHeader()
    {
        using var ms = new MemoryStream();
        WriteMinimalGguf(ms);
        ms.Position = 0;

        var model = GgufReader.Read(ms);

        Assert.Equal(3u, model.Version);
        Assert.Equal(0u, model.TensorCount);
        Assert.Equal(0u, model.MetadataKvCount);
    }

    [Fact]
    public void Read_WithMetadata_ParsesMetadataValues()
    {
        using var ms = new MemoryStream();
        var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);

        WriteMagicAndVersion(writer);
        writer.Write((ulong)0);
        writer.Write((ulong)1);

        WriteGgufString(writer, "general.architecture");
        writer.Write((uint)GgufValueType.String);
        WriteGgufString(writer, "phi2");

        writer.Flush();
        ms.Position = 0;

        var model = GgufReader.Read(ms);

        Assert.True(model.Metadata.TryGetValue("general.architecture", out var val));
        Assert.NotNull(val);
        Assert.Equal("phi2", val.Value);
    }

    private static void WriteMinimalGguf(Stream stream)
    {
        var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        WriteMagicAndVersion(writer);
        writer.Write((ulong)0);
        writer.Write((ulong)0);
        writer.Flush();
    }

    private static void WriteMagicAndVersion(BinaryWriter writer)
    {
        writer.Write(new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F' });
        writer.Write((uint)3);
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}