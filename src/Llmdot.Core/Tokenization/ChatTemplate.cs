using Llmdot.Loading;
using Llmdot.Tokenization.Jinja;

namespace Llmdot.Tokenization;

public sealed class ChatTemplate
{
    private readonly string _rawTemplate;
    private readonly string _bosToken;
    private readonly string _eosToken;

    public string RawTemplate => _rawTemplate;

    private ChatTemplate(string rawTemplate, string bosToken, string eosToken)
    {
        _rawTemplate = rawTemplate;
        _bosToken = bosToken;
        _eosToken = eosToken;
    }

    internal static ChatTemplate? FromGguf(GgufMetadata metadata, BpeTokenizer tokenizer)
    {
        var raw = metadata.GetOrDefault<string>("tokenizer.chat_template", null);
        if (raw is null) return null;

        var bosToken = tokenizer.Decode(tokenizer.BosTokenId);
        var eosToken = tokenizer.Decode(tokenizer.EosTokenId);

        return new ChatTemplate(raw, bosToken, eosToken);
    }

    public string Format(IReadOnlyList<ChatMessageEntry> messages)
    {
        var context = BuildContext(messages);

        try
        {
            return JinjaTemplate.Render(_rawTemplate, context);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to render chat template. Template: '{TruncateForError(_rawTemplate, 200)}'. Error: {ex.Message}", ex);
        }
    }

    private Dictionary<string, object?> BuildContext(IReadOnlyList<ChatMessageEntry> messages)
    {
        var messageList = messages.Select(m => (object?)new Dictionary<string, object?>
        {
            ["role"] = m.Role,
            ["content"] = m.Content,
        }).ToList();

        return new Dictionary<string, object?>
        {
            ["messages"] = messageList,
            ["bos_token"] = _bosToken,
            ["eos_token"] = _eosToken,
            ["add_generation_prompt"] = true,
            ["namespace"] = (Func<object?[], Dictionary<string, object?>, object?>)((args, kwargs) =>
            {
                var ns = new JinjaNamespace();
                foreach (var kv in kwargs)
                    ns[kv.Key] = kv.Value;
                return ns;
            }),
            ["keep_past_thinking"] = false,
            ["tools"] = null,
        };
    }

    private static string TruncateForError(string s, int maxLength) =>
        s.Length <= maxLength ? s : s[..maxLength] + "...";
}

public sealed class ChatMessageEntry
{
    public string Role { get; }
    public string Content { get; }

    public ChatMessageEntry(string role, string content)
    {
        Role = role;
        Content = content;
    }
}