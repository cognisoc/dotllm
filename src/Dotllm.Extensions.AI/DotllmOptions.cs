namespace Dotllm.Extensions.AI;

public sealed class DotllmOptions
{
    public string ModelPath { get; set; } = string.Empty;
    public int ContextLength { get; set; } = 4096;
    public int MaxTokens { get; set; } = 256;
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.95f;
    public float RepeatPenalty { get; set; } = 1.1f;
}