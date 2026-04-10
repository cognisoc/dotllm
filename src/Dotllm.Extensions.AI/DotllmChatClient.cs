using System.Runtime.CompilerServices;
using Dotllm.Inference;
using Dotllm.Sampling;
using Dotllm.Tokenization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace Dotllm.Extensions.AI;

public sealed class DotllmChatClient : IChatClient
{
    private readonly LoadedModel _model;
    private readonly InferenceEngine _engine;
    private readonly DotllmOptions _options;
    private readonly ChatClientMetadata _metadata;

    public DotllmChatClient(IOptions<DotllmOptions> options)
    {
        _options = options.Value;

        if (string.IsNullOrWhiteSpace(_options.ModelPath))
            throw new InvalidOperationException("DotllmOptions.ModelPath must be set.");

        var stream = File.OpenRead(_options.ModelPath);
        _model = LoadedModel.Load(stream);
        _engine = new InferenceEngine(_model);
        _metadata = new ChatClientMetadata(
            providerName: "dotllm",
            defaultModelId: _model.Config.Architecture);
    }

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = FormatChatMessages(messages);
        var tokens = _model.Tokenizer.Encode(prompt);

        var genOptions = ToGenerationOptions(options);

        var resultTokens = new List<int>();
        await foreach (var token in _engine.Generate(tokens, genOptions, cancellationToken))
            resultTokens.Add(token);

        var text = _model.Tokenizer.Decode(resultTokens);

        return new ChatResponse(new ChatMessage(ChatRole.Assistant, text))
        {
            ModelId = _model.Config.Architecture,
        };
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return StreamTokens(messages, options, cancellationToken);
    }

    private async IAsyncEnumerable<ChatResponseUpdate> StreamTokens(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var prompt = FormatChatMessages(messages);
        var tokens = _model.Tokenizer.Encode(prompt);

        var genOptions = ToGenerationOptions(options);

        await foreach (var token in _engine.Generate(tokens, genOptions, cancellationToken))
        {
            var text = _model.Tokenizer.Decode(token);
            yield return new ChatResponseUpdate(ChatRole.Assistant, text);
        }
    }

    public ChatClientMetadata Metadata => _metadata;

    public TService? GetService<TService>(object? serviceKey = null) where TService : class =>
        this as TService;

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceType == typeof(DotllmChatClient) ? this : null;

    public void Dispose() => _model.Dispose();

    private GenerationOptions ToGenerationOptions(ChatOptions? options) => new()
    {
        MaxTokens = options?.MaxOutputTokens ?? _options.MaxTokens,
        Sampling = new SamplingOptions
        {
            Temperature = options?.Temperature ?? _options.Temperature,
            TopK = _options.TopK,
            TopP = _options.TopP,
            RepeatPenalty = _options.RepeatPenalty,
        },
    };

    private string FormatChatMessages(IEnumerable<ChatMessage> messages)
    {
        var entries = messages
            .Where(m => !string.IsNullOrEmpty(m.Text))
            .Select(m => new ChatMessageEntry(ToTemplateRole(m.Role), m.Text!))
            .ToList();

        if (_model.ChatTemplate is not null)
            return _model.ChatTemplate.Format(entries);

        return FallbackFormat(entries);
    }

    private static string ToTemplateRole(ChatRole role) =>
        role == ChatRole.System ? "system" :
        role == ChatRole.Assistant ? "assistant" :
        role == ChatRole.User ? "user" :
        role.Value;

    private static string FallbackFormat(IReadOnlyList<ChatMessageEntry> messages)
    {
        var sb = new System.Text.StringBuilder();
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