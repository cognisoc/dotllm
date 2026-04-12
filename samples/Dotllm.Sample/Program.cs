using Dotllm.Inference;
using Dotllm.Tokenization;

namespace Dotllm.Sample;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: dotllm <model.gguf> [prompt]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  <model.gguf>  Path to a GGUF model file");
            Console.WriteLine("  [prompt]      Optional prompt text (interactive mode if omitted)");
            return 1;
        }

        var modelPath = args[0];
        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Error: Model file not found: {modelPath}");
            return 1;
        }

        Console.WriteLine($"Loading model from {modelPath}...");

        LoadedModel model;
        try
        {
            using var stream = File.OpenRead(modelPath);
            model = LoadedModel.Load(stream);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error loading model: {ex.Message}");
            return 1;
        }

        Console.WriteLine($"Model: {model.Capabilities.Architecture}");
        Console.WriteLine($"  Hidden:   {model.Config.HiddenSize}");
        Console.WriteLine($"  Layers:    {model.Config.LayerCount}");
        Console.WriteLine($"  Context:   {model.Config.ContextLength}");
        Console.WriteLine($"  Vocab:     {model.Config.VocabSize}");
        Console.WriteLine($"  Heads:     {model.Config.HeadCount}Q / {model.Config.HeadCountKv}KV");
        Console.WriteLine($"  Template:  {model.Capabilities.Template}");

        var session = new ChatSession(model);

        if (args.Length > 1)
        {
            var prompt = string.Join(' ', args[1..]);
            GenerateAndPrint(session, prompt);
        }
        else
        {
            RunInteractive(session);
        }

        model.Dispose();
        return 0;
    }

    private static void GenerateAndPrint(ChatSession session, string prompt)
    {
        Console.Write("> ");
        Console.WriteLine(prompt);
        Console.Write(" ");

        var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        foreach (var piece in session.GenerateAsync(prompt, cancellationToken: cts.Token).ToBlockingEnumerable())
            Console.Write(piece);

        Console.WriteLine();
    }

    private static void RunInteractive(ChatSession session)
    {
        Console.WriteLine("Interactive mode. Type 'quit' to exit, 'reset' to clear history.");
        Console.WriteLine();

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input))
                continue;

            if (input.Equals("quit", StringComparison.OrdinalIgnoreCase) ||
                input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            if (input.Equals("reset", StringComparison.OrdinalIgnoreCase))
            {
                session.Reset();
                Console.WriteLine("(history cleared)");
                continue;
            }

            Console.Write(" ");

            var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            foreach (var piece in session.GenerateAsync(input, new GenerationOptions { MaxTokens = 512 }, cts.Token).ToBlockingEnumerable())
                Console.Write(piece);

            Console.WriteLine();
            Console.WriteLine();
        }
    }
}