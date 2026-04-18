using Llmdot.Inference;

namespace Llmdot.Cli.Commands;

internal static class CompleteCommand
{
    public static async Task<int> RunAsync(string modelPath, string prompt, CommandOptions options, CancellationToken cancellationToken)
    {
        if (!File.Exists(modelPath))
        {
            CliOutput.WriteError($"Model file not found: {modelPath}");
            return 1;
        }

        using var stream = File.OpenRead(modelPath);
        using var model = LoadedModel.Load(stream);
        var engine = new InferenceEngine(model);

        var tokens = model.Tokenizer.Encode(prompt);
        if (model.Config.BosTokenId > 0)
            tokens = [model.Config.BosTokenId, .. tokens];

        var genOptions = options.ToGenerationOptions();

        await foreach (var tokenId in engine.Generate(tokens, genOptions, cancellationToken))
        {
            var piece = model.Tokenizer.Decode([tokenId]);
            Console.Write(piece);
        }

        Console.WriteLine();
        return 0;
    }
}
