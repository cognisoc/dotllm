using Dotllm.Extensions.AI;
using Dotllm.Inference;

namespace Dotllm.Cli.Commands;

internal static class ChatCommand
{
    public static async Task<int> RunAsync(string modelPath, string? initialPrompt, CommandOptions options, CancellationToken cancellationToken)
    {
        if (!File.Exists(modelPath))
        {
            CliOutput.WriteError($"Model file not found: {modelPath}");
            return 1;
        }

        using var stream = File.OpenRead(modelPath);
        using var model = LoadedModel.Load(stream);

        if (model.ChatTemplate is null)
        {
            CliOutput.WriteError("Model does not have a chat template. Use 'complete' command instead.");
            return 1;
        }

        var session = new ChatSession(model);
        var genOptions = options.ToGenerationOptions();

        if (initialPrompt is not null)
        {
            // One-shot mode
            await foreach (var piece in session.GenerateAsync(initialPrompt, genOptions, cancellationToken))
                Console.Write(piece);
            Console.WriteLine();
            return 0;
        }

        // Interactive mode
        Console.WriteLine("Chat session started. Type 'exit' or press Ctrl+C to quit.");
        Console.WriteLine();

        while (!cancellationToken.IsCancellationRequested)
        {
            Console.Write("> ");
            var input = Console.ReadLine();

            if (input is null || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            if (string.IsNullOrWhiteSpace(input))
                continue;

            await foreach (var piece in session.GenerateAsync(input, genOptions, cancellationToken))
                Console.Write(piece);

            Console.WriteLine();
            Console.WriteLine();
        }

        return 0;
    }
}
