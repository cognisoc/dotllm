using System.Globalization;
using System.Text;
using Dotllm.Loading;

namespace Dotllm.Tokenization;

public sealed class BpeTokenizer
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;
    private readonly Dictionary<(string, string), int> _mergeRanks;

    public int VocabSize => _tokenToId.Count;
    public int BosTokenId { get; }
    public int EosTokenId { get; }

    internal BpeTokenizer(
        string[] tokens,
        float[] scores,
        string[] merges,
        int bosTokenId,
        int eosTokenId)
    {
        _tokenToId = new Dictionary<string, int>(tokens.Length);
        _idToToken = new Dictionary<int, string>(tokens.Length);
        _mergeRanks = new Dictionary<(string, string), int>(merges.Length);

        for (var i = 0; i < tokens.Length; i++)
        {
            _tokenToId[tokens[i]] = i;
            _idToToken[i] = tokens[i];
        }

        for (var i = 0; i < merges.Length; i++)
        {
            var parts = merges[i].Split(' ', 2);
            if (parts.Length == 2)
                _mergeRanks[(parts[0], parts[1])] = i;
        }

        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
    }

    internal static BpeTokenizer FromGguf(GgufMetadata metadata)
    {
        var tokenStrings = GetArray<string>(metadata, "tokenizer.ggml.tokens");
        var tokenScores = GetArray<float>(metadata, "tokenizer.ggml.scores");
        var mergeStrings = GetArray<string>(metadata, "tokenizer.ggml.merges");

        var bosTokenId = GetInt(metadata, "tokenizer.ggml.bos_token_id", 1);
        var eosTokenId = GetInt(metadata, "tokenizer.ggml.eos_token_id", 2);

        return new BpeTokenizer(tokenStrings, tokenScores, mergeStrings, bosTokenId, eosTokenId);
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
            return [];

        var result = new List<int>();
        var i = 0;

        while (i < text.Length)
        {
            var bestId = -1;
            var bestLen = 0;

            for (var len = Math.Min(64, text.Length - i); len >= 1; len--)
            {
                var piece = text.Substring(i, len);
                if (_tokenToId.TryGetValue(piece, out var id))
                {
                    bestId = id;
                    bestLen = len;
                    break;
                }
            }

            if (bestId >= 0)
            {
                result.Add(bestId);
                i += bestLen;
                continue;
            }

            var ch = text[i];
            var chStr = ch.ToString();
            if (_tokenToId.TryGetValue(chStr, out var charId))
            {
                result.Add(charId);
                i++;
                continue;
            }

            var bytes = Encoding.UTF8.GetBytes(chStr);
            foreach (var b in bytes)
            {
                var byteToken = $"<0x{b:X2}>";
                if (_tokenToId.TryGetValue(byteToken, out var byteId))
                    result.Add(byteId);
                else
                    result.Add(UnknownTokenId());
            }

            i++;
        }

        if (_mergeRanks.Count > 0)
            result = ApplyMerges(result);

        return result.ToArray();
    }

    private List<int> ApplyMerges(List<int> tokens)
    {
        if (tokens.Count <= 1)
            return tokens;

        var symbols = new List<string>(tokens.Count);
        foreach (var id in tokens)
            symbols.Add(_idToToken.TryGetValue(id, out var t) ? t : UnknownTokenId().ToString(CultureInfo.InvariantCulture));

        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIdx = -1;

            for (var j = 0; j < symbols.Count - 1; j++)
            {
                if (_mergeRanks.TryGetValue((symbols[j], symbols[j + 1]), out var rank))
                {
                    if (rank < bestRank)
                    {
                        bestRank = rank;
                        bestIdx = j;
                    }
                }
            }

            if (bestIdx < 0)
                break;

            var merged = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
            symbols[bestIdx] = merged;
        }

        var result = new List<int>(symbols.Count);
        foreach (var sym in symbols)
        {
            if (_tokenToId.TryGetValue(sym, out var id))
                result.Add(id);
            else
                result.Add(UnknownTokenId());
        }

        return result;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (_idToToken.TryGetValue(id, out var token))
                sb.Append(token);
        }

        return CleanDecoded(sb.ToString());
    }

    public string Decode(int id) =>
        _idToToken.TryGetValue(id, out var token) ? CleanDecoded(token) : string.Empty;

    public bool TryGetTokenId(string token, out int id) =>
        _tokenToId.TryGetValue(token, out id);

    private int UnknownTokenId() =>
        _tokenToId.TryGetValue("<unk>", out var id) ? id : 0;

    private static string CleanDecoded(string text)
    {
        var sb = new StringBuilder();
        var i = 0;

        while (i < text.Length)
        {
            if (i + 5 < text.Length &&
                text[i] == '<' && text[i + 1] == '0' && text[i + 2] == 'x' &&
                text[i + 5] == '>')
            {
                var hexStr = text.Substring(i + 3, 2);
                if (byte.TryParse(hexStr, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var b))
                {
                    sb.Append((char)b);
                    i += 6;
                    continue;
                }
            }

            sb.Append(text[i]);
            i++;
        }

        return sb.ToString();
    }

    private static T[] GetArray<T>(GgufMetadata metadata, string key) where T : notnull
    {
        if (metadata.TryGetValue(key, out var val) && val is not null && val.Value is object[] arr)
        {
            var result = new T[arr.Length];
            for (var i = 0; i < arr.Length; i++)
                result[i] = (T)Convert.ChangeType(arr[i], typeof(T), CultureInfo.InvariantCulture);
            return result;
        }
        return [];
    }

    private static int GetInt(GgufMetadata metadata, string key, int defaultValue)
    {
        if (metadata.TryGetValue(key, out var val) && val is not null)
            return Convert.ToInt32(val.Value, CultureInfo.InvariantCulture);
        return defaultValue;
    }
}