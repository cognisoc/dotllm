namespace Dotllm.Sampling;

/// <summary>
/// Controls token sampling parameters: temperature, top-k, top-p, and repeat penalty.
/// Use <see cref="Seed"/> >= 0 for deterministic output, or -1 for random sampling.
/// </summary>
public sealed class SamplingOptions
{
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    public float Temperature { get; init; } = 0.8f;
    public float RepeatPenalty { get; init; } = 1.1f;
    public int RepeatPenaltyWindowSize { get; init; } = 64;
    public int Seed { get; init; } = -1;
}