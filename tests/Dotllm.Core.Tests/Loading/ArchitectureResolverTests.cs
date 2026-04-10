using System.Text;
using Dotllm.Loading;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Loading;

public class ArchitectureResolverTests
{
    [Fact]
    public void Resolve_MissingArchitecture_Throws()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, []);
        ms.Position = 0;
        var model = GgufReader.Read(ms);

        Assert.Throws<InvalidDataException>(() => ArchitectureResolver.Resolve(model));
    }

    [Fact]
    public void Resolve_UnsupportedArchitecture_Throws()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [("general.architecture", GgufValueType.String, "fancy_new_model")]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);

        var ex = Assert.Throws<InvalidDataException>(() => ArchitectureResolver.Resolve(model));
        Assert.Contains("Unsupported GGUF architecture", ex.Message);
    }

    [Fact]
    public void Resolve_LlamaArchitecture_ResolvesLlamaLikeTemplate()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 4096),
            ("llama.block_count", GgufValueType.UInt32, 32),
            ("llama.context_length", GgufValueType.UInt32, 4096),
            ("llama.feed_forward_length", GgufValueType.UInt32, 11008),
            ("llama.attention.head_count", GgufValueType.UInt32, 32),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 32),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal("llama", config.Architecture);
        Assert.Equal(ExecutionTemplate.LlamaLike, config.Template);
        Assert.Equal(4096, config.HiddenSize);
        Assert.Equal(32, config.LayerCount);
        Assert.Equal(AttentionType.MHA, config.AttentionType);
        Assert.Equal(NormType.RmsNorm, config.NormType);
        Assert.Equal(1e-5f, config.NormEpsilon);
    }

    [Fact]
    public void Resolve_Phi2Architecture_ResolvesGptNeoXLikeTemplate()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "phi2"),
            ("phi2.embedding_length", GgufValueType.UInt32, 2560),
            ("phi2.block_count", GgufValueType.UInt32, 32),
            ("phi2.context_length", GgufValueType.UInt32, 2048),
            ("phi2.feed_forward_length", GgufValueType.UInt32, 10240),
            ("phi2.attention.head_count", GgufValueType.UInt32, 32),
            ("phi2.attention.head_count_kv", GgufValueType.UInt32, 32),
            ("phi2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("phi2.use_parallel_residual", GgufValueType.Bool, true),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal("phi2", config.Architecture);
        Assert.Equal(ExecutionTemplate.GptNeoXLike, config.Template);
        Assert.True(config.ParallelResidual);
    }

    [Fact]
    public void Resolve_GemmaArchitecture_ResolvesGemmaLikeTemplate()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "gemma"),
            ("gemma.embedding_length", GgufValueType.UInt32, 2048),
            ("gemma.block_count", GgufValueType.UInt32, 18),
            ("gemma.context_length", GgufValueType.UInt32, 8192),
            ("gemma.feed_forward_length", GgufValueType.UInt32, 16384),
            ("gemma.attention.head_count", GgufValueType.UInt32, 8),
            ("gemma.attention.head_count_kv", GgufValueType.UInt32, 1),
            ("gemma.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-6f),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal("gemma", config.Architecture);
        Assert.Equal(ExecutionTemplate.GemmaLike, config.Template);
        Assert.Equal(AttentionType.MQA, config.AttentionType);
        Assert.Equal(MathF.Sqrt(2048), config.EmbeddingScale);
    }

    [Fact]
    public void Resolve_GqaArchitecture_DetectsGqa()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 4096),
            ("llama.block_count", GgufValueType.UInt32, 32),
            ("llama.context_length", GgufValueType.UInt32, 8192),
            ("llama.feed_forward_length", GgufValueType.UInt32, 14336),
            ("llama.attention.head_count", GgufValueType.UInt32, 32),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 8),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal(AttentionType.GQA, config.AttentionType);
        Assert.Equal(8, config.HeadCountKv);
    }

    [Fact]
    public void Resolve_Lfm2Architecture_ResolvesLfm2LikeTemplate()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "lfm2"),
            ("lfm2.embedding_length", GgufValueType.UInt32, 2048),
            ("lfm2.block_count", GgufValueType.UInt32, 16),
            ("lfm2.context_length", GgufValueType.UInt32, 32768),
            ("lfm2.feed_forward_length", GgufValueType.UInt32, 6656),
            ("lfm2.attention.head_count", GgufValueType.UInt32, 16),
            ("lfm2.attention.head_count_kv", GgufValueType.UInt32, 8),
            ("lfm2.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("lfm2.conv_L_cache", GgufValueType.UInt32, 3),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal("lfm2", config.Architecture);
        Assert.Equal(ExecutionTemplate.Lfm2Like, config.Template);
        Assert.True(config.HasConvLayers);
        Assert.Equal(3, config.ConvKernelSize);
        Assert.Equal(16, config.LayerTypes.Length);
        Assert.Equal(AttentionType.GQA, config.AttentionType);
    }

    [Fact]
    public void Resolve_NonLfm2Architecture_HasNoConvLayers()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 4096),
            ("llama.block_count", GgufValueType.UInt32, 32),
            ("llama.context_length", GgufValueType.UInt32, 4096),
            ("llama.feed_forward_length", GgufValueType.UInt32, 11008),
            ("llama.attention.head_count", GgufValueType.UInt32, 32),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 32),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.False(config.HasConvLayers);
        Assert.Equal(0, config.ConvKernelSize);
        Assert.All(config.LayerTypes, lt => Assert.Equal(LayerType.Attention, lt));
    }

    private static void WriteGguf(Stream stream, (string key, GgufValueType type, object value)[] metadata)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F' });
        writer.Write((uint)3);
        writer.Write((ulong)0);
        writer.Write((ulong)metadata.Length);

        foreach (var (key, type, value) in metadata)
        {
            WriteGgufString(writer, key);
            writer.Write((uint)type);
            WriteMetadataValue(writer, type, value);
        }

        writer.Flush();
    }

    private static void WriteMetadataValue(BinaryWriter writer, GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.UInt32:
                WriteU32LE(writer, (uint)(int)value);
                break;
            case GgufValueType.Float32:
                WriteF32LE(writer, (float)value);
                break;
            case GgufValueType.String:
                WriteGgufString(writer, (string)value);
                break;
            case GgufValueType.Bool:
                WriteU32LE(writer, (bool)value ? 1u : 0u);
                break;
            default:
                throw new NotSupportedException($"Test helper doesn't support GGUF value type: {type}");
        }
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        WriteU64LE(writer, (ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void WriteU32LE(BinaryWriter writer, uint value)
    {
        var buf = new byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
        writer.Write(buf);
    }

    private static void WriteU64LE(BinaryWriter writer, ulong value)
    {
        var buf = new byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(buf, value);
        writer.Write(buf);
    }

    private static void WriteF32LE(BinaryWriter writer, float value)
    {
        var buf = new byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(buf, value);
        writer.Write(buf);
    }
}