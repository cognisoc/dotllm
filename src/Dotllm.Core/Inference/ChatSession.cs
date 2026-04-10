using System.Runtime.CompilerServices;
using System.Text;
using Dotllm.Loading;
using Dotllm.Models;
using Dotllm.Sampling;
using Dotllm.Tensors;
using Dotllm.Tensors.Numeric;
using Dotllm.Tokenization;

namespace Dotllm.Inference;

public sealed class ChatSession
{
    private readonly InferenceEngine _engine;
    private readonly BpeTokenizer _tokenizer;
    private readonly TransformerConfig _config;
    private readonly List<int> _history;

    public ChatSession(LoadedModel model)
    {
        _engine = new InferenceEngine(model);
        _tokenizer = model.Tokenizer;
        _config = model.Config;
        _history = new List<int>();
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        GenerationOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new GenerationOptions();

        var promptTokens = _tokenizer.Encode(prompt);
        var inputTokens = new List<int>();

        if (_history.Count == 0 && _config.BosTokenId > 0)
            inputTokens.Add(_config.BosTokenId);

        inputTokens.AddRange(promptTokens);

        var tokenBuffer = new List<int>(inputTokens);
        var sb = new StringBuilder();
        var prevText = string.Empty;

        await foreach (var tokenId in _engine.Generate(inputTokens.ToArray(), options, cancellationToken))
        {
            _history.Add(tokenId);
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
    }

    public void Reset() => _history.Clear();
}