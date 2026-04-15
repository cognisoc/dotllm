using Dotllm.Inference;
using Dotllm.Models;
using Dotllm.Tokenization;

namespace Dotllm.Sample;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: dotllm <model.gguf> [prompt]");
            return 1;
        }

        if (!File.Exists(args[0]))
        {
            Console.Error.WriteLine($"Error: Model file not found: {args[0]}");
            return 1;
        }

        Console.WriteLine($"Loading model from {args[0]}...");

        LoadedModel model;
        try
        {
            using var stream = File.OpenRead(args[0]);
            model = LoadedModel.Load(stream);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error loading model: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }

        Console.WriteLine($"Model: {model.Capabilities.Architecture}");
        Console.WriteLine($"  Hidden:   {model.Config.HiddenSize}");
        Console.WriteLine($"  Layers:    {model.Config.LayerCount}");
        Console.WriteLine($"  Context:   {model.Config.ContextLength}");
        Console.WriteLine($"  Vocab:     {model.Config.VocabSize}");
        Console.WriteLine($"  Heads:     {model.Config.HeadCount}Q / {model.Config.HeadCountKv}KV");
        Console.WriteLine($"  Template:  {model.Capabilities.Template}");
        Console.WriteLine($"  HasConv:   {model.Config.HasConvLayers}");
        Console.WriteLine($"  ConvKernelSize: {model.Config.ConvKernelSize}");
        if (model.Config.HeadCountKvPerLayer.Length > 0)
            Console.WriteLine($"  HeadCountKvPerLayer: [{string.Join(", ", model.Config.HeadCountKvPerLayer)}]");
        Console.WriteLine($"  LayerTypes: [{string.Join(", ", model.Config.LayerTypes)}]");

        var engine = new InferenceEngine(model);

        if (args.Length > 1)
        {
            var prompt = string.Join(' ', args[1..]);
            Console.Write("> ");
            Console.WriteLine(prompt);

            var tokens = PreparePromptTokens(model, prompt);
            Console.WriteLine($"  Encoded {tokens.Length} tokens");
            RunInference(engine, model, tokens, 64);
        }
        else
        {
            Console.WriteLine("Interactive mode. Type 'quit' to exit.");
            Console.WriteLine();

            while (true)
            {
                Console.Write("> ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input)) continue;
                if (input.Equals("quit", StringComparison.OrdinalIgnoreCase)) break;

                var tokens = PreparePromptTokens(model, input);
                Console.WriteLine($"  Encoded {tokens.Length} tokens");
                RunInference(engine, model, tokens, 128);
                Console.WriteLine();
            }
        }

        model.Dispose();
        return 0;
    }

    private static void RunInference(InferenceEngine engine, LoadedModel model, int[] tokens, int maxTokens)
    {
        var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        foreach (var tokenId in engine.Generate(tokens, new GenerationOptions { MaxTokens = maxTokens }, cts.Token).ToBlockingEnumerable())
            Console.Write(model.Tokenizer.Decode([tokenId]));

        Console.WriteLine();
    }

    private static int[] PreparePromptTokens(LoadedModel model, string prompt)
    {
        var tokens = model.Tokenizer.Encode(prompt);
        var bosTokenId = model.Config.BosTokenId;

        if (bosTokenId > 0 && (tokens.Length == 0 || tokens[0] != bosTokenId))
            return [bosTokenId, .. tokens];

        return tokens;
    }
}
