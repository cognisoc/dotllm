using System.Text;
using Llmdot.Loading;
using Llmdot.Tokenization;
using Xunit;

namespace Llmdot.Core.Tests.Tokenization;

public class BpeTokenizerTests
{
    [Fact]
    public void FromGguf_ParsesTokensAndMerges()
    {
        using var ms = new MemoryStream();
        WriteGguf(ms, [
            ("general.architecture", GgufValueType.String, "llama"),
            ("llama.embedding_length", GgufValueType.UInt32, 32),
            ("llama.block_count", GgufValueType.UInt32, 1),
            ("llama.context_length", GgufValueType.UInt32, 128),
            ("llama.feed_forward_length", GgufValueType.UInt32, 64),
            ("llama.attention.head_count", GgufValueType.UInt32, 4),
            ("llama.attention.head_count_kv", GgufValueType.UInt32, 4),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
        ]);
        ms.Position = 0;
        var model = GgufReader.Read(ms);

        var tokenizer = BpeTokenizer.FromGguf(model.Metadata);

        Assert.Equal(0, tokenizer.VocabSize);
    }

    [Fact]
    public void Encode_Decode_Roundtrip()
    {
        var tokens = new[] { "<unk>", "▁hello", "▁world", "▁", "h", "e", "l", "o", "▁w", "r", "d" };
        var scores = new float[] { 0f, -1f, -2f, -3f, -4f, -5f, -6f, -7f, -8f, -9f, -10f };
        var merges = new[] { "▁ h", "▁hello ▁world" };

        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 0, eosTokenId: 0);

        var encoded = tokenizer.Encode("▁hello");
        Assert.NotEmpty(encoded);
        Assert.Equal(1, encoded[0]);

        var decoded = tokenizer.Decode(encoded);
        Assert.Equal("▁hello", decoded);
    }

    [Fact]
    public void Decode_SingleToken()
    {
        var tokens = new[] { "<unk>", "▁hello", "▁world" };
        var scores = new float[] { 0f, -1f, -2f };
        var merges = Array.Empty<string>();

        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 0, eosTokenId: 0);

        var decoded = tokenizer.Decode(1);
        Assert.Equal("▁hello", decoded);
    }

    [Fact]
    public void BosEosTokenIds_SetCorrectly()
    {
        var tokens = new[] { "<pad>", "<bos>", "<eos>" };
        var scores = new float[] { 0f, 0f, 0f };
        var merges = Array.Empty<string>();

        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 1, eosTokenId: 2);

        Assert.Equal(1, tokenizer.BosTokenId);
        Assert.Equal(2, tokenizer.EosTokenId);
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        var tokens = new[] { "<unk>" };
        var scores = new float[] { 0f };
        var merges = Array.Empty<string>();

        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 0, eosTokenId: 0);
        var encoded = tokenizer.Encode("");

        Assert.Empty(encoded);
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
                throw new NotSupportedException($"Test helper: {type}");
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