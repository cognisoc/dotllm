using System.Runtime.CompilerServices;
using System.Text;
using Dotllm.Models;
using Dotllm.Sampling;
using Dotllm.Tokenization;

namespace Dotllm.Inference;

public sealed class ChatSession
{
    private readonly InferenceEngine _engine;
    private readonly BpeTokenizer _tokenizer;
    private readonly TransformerConfig _config;
    private readonly ChatTemplate? _chatTemplate;
    private readonly List<ChatMessageEntry> _history = [];

    public ChatSession(LoadedModel model)
    {
        _engine = new InferenceEngine(model);
        _tokenizer = model.Tokenizer;
        _config = model.Config;
        _chatTemplate = model.ChatTemplate;
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new GenerationOptions();

        _history.Add(new ChatMessageEntry("user", prompt));

        var formattedPrompt = _chatTemplate is not null
            ? _chatTemplate.Format(_history)
            : FallbackFormat(_history);

        var promptTokens = _tokenizer.Encode(formattedPrompt);

        if (_history.Count == 1 && _config.BosTokenId > 0)
            promptTokens = [_config.BosTokenId, .. promptTokens];

        var sb = new StringBuilder();
        var prevText = string.Empty;

        await foreach (var tokenId in _engine.Generate(promptTokens, options, cancellationToken))
        {
            var piece = _tokenizer.Decode(tokenId);
            sb.Append(piece);
            var currentText = sb.ToString();

            if (currentText.Length > prevText.Length)
            {
                var newText = currentText[prevText.Length..];
                prevText = currentText;
                yield return newText;
            }
        }

        var assistantText = sb.ToString();
        if (!string.IsNullOrEmpty(assistantText))
            _history.Add(new ChatMessageEntry("assistant", assistantText));
    }

    public void Reset() => _history.Clear();

    private static string FallbackFormat(IReadOnlyList<ChatMessageEntry> messages)
    {
        var sb = new StringBuilder();
        foreach (var msg in messages)
        {
            sb.Append('<');
            sb.Append(msg.Role);
            sb.Append('>');
            sb.AppendLine(msg.Content);
        }

        sb.Append("<assistant>");
        return sb.ToString();
    }
}