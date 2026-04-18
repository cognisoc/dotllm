namespace Llmdot.Cli.Commands;

internal sealed class CommandOptions
{
    public int MaxTokens { get; set; } = 256;
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.95f;
    public float RepeatPenalty { get; set; } = 1.1f;
    public int Seed { get; set; } = -1;

    public static CommandOptions Parse(string[] args)
    {
        var options = new CommandOptions();
        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--max-tokens" when i + 1 < args.Length:
                    options.MaxTokens = int.Parse(args[++i]);
                    break;
                case "--temperature" when i + 1 < args.Length:
                    options.Temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--top-k" when i + 1 < args.Length:
                    options.TopK = int.Parse(args[++i]);
                    break;
                case "--top-p" when i + 1 < args.Length:
                    options.TopP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--repeat-penalty" when i + 1 < args.Length:
                    options.RepeatPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "--seed" when i + 1 < args.Length:
                    options.Seed = int.Parse(args[++i]);
                    break;
            }
        }
        return options;
    }

    public Llmdot.Sampling.SamplingOptions ToSamplingOptions() => new()
    {
        TopK = TopK,
        TopP = TopP,
        Temperature = Temperature,
        RepeatPenalty = RepeatPenalty,
        Seed = Seed,
    };

    public Llmdot.Inference.GenerationOptions ToGenerationOptions() => new()
    {
        MaxTokens = MaxTokens,
        Sampling = ToSamplingOptions(),
    };
}
