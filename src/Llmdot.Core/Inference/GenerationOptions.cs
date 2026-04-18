namespace Llmdot.Inference;

/// <summary>
/// Configuration for a single generation call: max tokens, sampling parameters, and stop sequences.
/// </summary>
public sealed class GenerationOptions
{
    public int MaxTokens { get; init; } = 256;
    public Sampling.SamplingOptions Sampling { get; init; } = new();
    public string? StopSequence { get; init; }
    public string[] StopSequences { get; init; } = [];
}