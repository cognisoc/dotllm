namespace Dotllm.Inference;

public sealed class GenerationOptions
{
    public int MaxTokens { get; init; } = 256;
    public Sampling.SamplingOptions Sampling { get; init; } = new();
    public string? StopSequence { get; init; }
}