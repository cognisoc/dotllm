namespace Dotllm.Inference;

public sealed class GenerationOptionsBuilder
{
    private int _maxTokens = 256;
    private float _temperature = 0.8f;
    private int _topK = 40;
    private float _topP = 0.95f;
    private float _repeatPenalty = 1.1f;
    private int _seed = -1;
    private readonly List<string> _stopSequences = [];

    public GenerationOptionsBuilder WithMaxTokens(int maxTokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokens);
        _maxTokens = maxTokens;
        return this;
    }

    public GenerationOptionsBuilder WithTemperature(float temperature)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(temperature);
        _temperature = temperature;
        return this;
    }

    public GenerationOptionsBuilder WithTopK(int topK)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(topK);
        _topK = topK;
        return this;
    }

    public GenerationOptionsBuilder WithTopP(float topP)
    {
        if (topP < 0f || topP > 1f)
            throw new ArgumentOutOfRangeException(nameof(topP), "TopP must be between 0.0 and 1.0");
        _topP = topP;
        return this;
    }

    public GenerationOptionsBuilder WithRepeatPenalty(float penalty)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(penalty);
        _repeatPenalty = penalty;
        return this;
    }

    public GenerationOptionsBuilder WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    public GenerationOptionsBuilder WithStopSequence(string stopSequence)
    {
        _stopSequences.Add(stopSequence);
        return this;
    }

    public GenerationOptions Build() => new()
    {
        MaxTokens = _maxTokens,
        Sampling = new Sampling.SamplingOptions
        {
            Temperature = _temperature,
            TopK = _topK,
            TopP = _topP,
            RepeatPenalty = _repeatPenalty,
            Seed = _seed,
        },
        StopSequences = [.. _stopSequences],
    };
}
