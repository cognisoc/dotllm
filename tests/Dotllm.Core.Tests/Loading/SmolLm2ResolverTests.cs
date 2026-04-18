using System.Text;
using Dotllm.Loading;
using Dotllm.Models;
using Xunit;

namespace Dotllm.Core.Tests.Loading;

public class SmolLm2ResolverTests
{
    [Fact]
    public void Resolve_SmolLm2_135M_Config()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 576),
            ("llama.block_count", GgufValueType.UInt32, 30),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 1536),
            ("llama.attention.head_count", GgufValueType.UInt32, 9),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 3),
            ("llama.attention.key_length", GgufValueType.UInt32, 64),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("llama.rope.freq_base", GgufValueType.Float32, 10000f),
        ], tensorNames: ["blk.0.ffn_gate.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal("llama", config.Architecture);
        Assert.Equal(ExecutionTemplate.LlamaLike, config.Template);
        Assert.Equal(576, config.HiddenSize);
        Assert.Equal(30, config.LayerCount);
        Assert.Equal(2048, config.ContextLength);
        Assert.Equal(1536, config.FfnDim);
        Assert.Equal(9, config.HeadCount);
        Assert.Equal(3, config.HeadCountKv);
        Assert.Equal(64, config.HeadDim);
        Assert.Equal(AttentionType.GQA, config.AttentionType);
        Assert.Equal(FfnType.SwiGLU, config.FfnType);
        Assert.Equal(NormType.RmsNorm, config.NormType);
        Assert.Equal(10000f, config.RopeFreqBase);
        Assert.True(config.TiedEmbeddings);
    }

    [Fact]
    public void Resolve_SmolLm2_360M_Config()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 960),
            ("llama.block_count", GgufValueType.UInt32, 32),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 2560),
            ("llama.attention.head_count", GgufValueType.UInt32, 15),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 5),
            ("llama.attention.key_length", GgufValueType.UInt32, 64),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ], tensorNames: ["blk.0.ffn_gate.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal(960, config.HiddenSize);
        Assert.Equal(32, config.LayerCount);
        Assert.Equal(2560, config.FfnDim);
        Assert.Equal(15, config.HeadCount);
        Assert.Equal(5, config.HeadCountKv);
        Assert.Equal(AttentionType.GQA, config.AttentionType);
    }

    [Fact]
    public void Resolve_SmolLm2_1_7B_Config()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 2048),
            ("llama.block_count", GgufValueType.UInt32, 24),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 8192),
            ("llama.attention.head_count", GgufValueType.UInt32, 32),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 32),
            ("llama.attention.key_length", GgufValueType.UInt32, 64),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ], tensorNames: ["blk.0.ffn_gate.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal(2048, config.HiddenSize);
        Assert.Equal(24, config.LayerCount);
        Assert.Equal(8192, config.FfnDim);
        Assert.Equal(32, config.HeadCount);
        Assert.Equal(32, config.HeadCountKv);
        Assert.Equal(AttentionType.MHA, config.AttentionType);
    }

    [Fact]
    public void Resolve_SmolLm2_TiedEmbeddings_NoOutputWeight()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 576),
            ("llama.block_count", GgufValueType.UInt32, 30),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 1536),
            ("llama.attention.head_count", GgufValueType.UInt32, 9),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 3),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ], tensorNames: ["token_embd.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.True(config.TiedEmbeddings);
    }

    [Fact]
    public void Resolve_SmolLm2_WithSeparateOutputWeight()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 576),
            ("llama.block_count", GgufValueType.UInt32, 30),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 1536),
            ("llama.attention.head_count", GgufValueType.UInt32, 9),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 3),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ], tensorNames: ["token_embd.weight", "output.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.False(config.TiedEmbeddings);
    }

    [Fact]
    public void Resolve_SmolLm2_SwiGLU_DetectedFromGateTensor()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 576),
            ("llama.block_count", GgufValueType.UInt32, 30),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 1536),
            ("llama.attention.head_count", GgufValueType.UInt32, 9),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 3),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ], tensorNames: ["blk.0.ffn_gate.weight"]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal(FfnType.SwiGLU, config.FfnType);
    }

    [Fact]
    public void Resolve_SmolLm2_RopeParameters()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 576),
            ("llama.block_count", GgufValueType.UInt32, 30),
            ("llama.context_length", GgufValueType.UInt32, 2048),
            ("llama.feed_forward_length", GgufValueType.UInt32, 1536),
            ("llama.attention.head_count", GgufValueType.UInt32, 9),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 3),
            ("llama.attention.key_length", GgufValueType.UInt32, 64),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("llama.rope.freq_base", GgufValueType.Float32, 10000f),
            ("llama.rope.dimension_count", GgufValueType.UInt32, 64),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);
        var config = ArchitectureResolver.Resolve(model);

        Assert.Equal(10000f, config.RopeFreqBase);
        Assert.Equal(64, config.RopeDimensionCount);
    }

    // GGUF writing helpers
    private static void WriteGguf(Stream stream, (string key, GgufValueType type, object value)[] metadata, string[]? tensorNames = null)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var tensors = tensorNames ?? [];

        writer.Write(new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F' });
        writer.Write((uint)3); // version
        writer.Write((ulong)tensors.Length); // tensor count
        writer.Write((ulong)metadata.Length); // metadata kv count

        foreach (var (key, type, value) in metadata)
        {
            WriteGgufString(writer, key);
            writer.Write((uint)type);
            WriteMetadataValue(writer, type, value);
        }

        // Write tensor infos
        foreach (var name in tensors)
        {
            WriteGgufString(writer, name);
            WriteU32LE(writer, 1); // n_dimensions
            WriteU64LE(writer, 1); // dim[0] - dimensions are read as uint64
            WriteU32LE(writer, 0); // type = F32
            WriteU64LE(writer, 0); // offset
        }

        // Pad to alignment boundary
        var pos = stream.Position;
        var align = 32;
        var padding = (align - (int)(pos % align)) % align;
        writer.Write(new byte[padding]);

        // Write minimal tensor data
        foreach (var _ in tensors)
            writer.Write(0f);

        writer.Flush();
    }

    private static void WriteMetadataValue(BinaryWriter writer, GgufValueType type, object value)
    {
        switch (type)
        {
            case GgufValueType.UInt32: WriteU32LE(writer, (uint)(int)value); break;
            case GgufValueType.Float32: WriteF32LE(writer, (float)value); break;
            case GgufValueType.String: WriteGgufString(writer, (string)value); break;
            case GgufValueType.Bool: WriteU32LE(writer, (bool)value ? 1u : 0u); break;
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
