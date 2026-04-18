using Llmdot.Inference;
using Llmdot.Models;
using Llmdot.Tokenization;

namespace Llmdot.Sample;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: llmdot <model.gguf> [prompt] [--raw] [--chat]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --raw   Force raw text completion mode (no chat template)");
            Console.WriteLine("  --chat  Force chat mode (uses chat template or fallback format)");
            Console.WriteLine();
            Console.WriteLine("By default, chat mode is used when the model has a chat template,");
            Console.WriteLine("and raw mode is used otherwise.");
            return 1;
        }

        var modelPath = args[0];
        var forceChat = false;
        var forceRaw = false;
        var promptArgs = new List<string>();

        for (var i = 1; i < args.Length; i++)
        {
            if (args[i] == "--raw") forceRaw = true;
            else if (args[i] == "--chat") forceChat = true;
            else promptArgs.Add(args[i]);
        }

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
        Console.WriteLine($"  Chat template: {(model.ChatTemplate is not null ? "yes" : "no")}");
        Console.WriteLine($"  QDim: {model.Config.QDim}, KvDim: {model.Config.KvDim}, HeadDim: {model.Config.HeadDim}");
        Console.WriteLine($"  FfnDim: {model.Config.FfnDim}, FfnType: {model.Config.FfnType}");
        Console.WriteLine($"  TiedEmbeddings: {model.Config.TiedEmbeddings}, EmbeddingScale: {model.Config.EmbeddingScale}");
        Console.WriteLine($"  RopeFreqBase: {model.Config.RopeFreqBase}, RopeDimCount: {model.Config.RopeDimensionCount}");
        Console.WriteLine($"  BOS: {model.Config.BosTokenId}, EOS: {model.Config.EosTokenId}");
        if (model.Config.HeadCountKvPerLayer.Count > 0)
            Console.WriteLine($"  HeadCountKvPerLayer: [{string.Join(", ", model.Config.HeadCountKvPerLayer)}]");

        var useChat = forceChat || (!forceRaw && model.ChatTemplate is not null);
        Console.WriteLine($"  Mode:      {(useChat ? "chat" : "raw completion")}");
        Console.WriteLine();

        if (useChat)
            RunChatMode(model, promptArgs);
        else
            RunRawMode(model, promptArgs);

        model.Dispose();
        return 0;
    }

    private static void RunChatMode(LoadedModel model, List<string> promptArgs)
    {
        var session = new ChatSession(model);
        var maxTokens = 256;

        if (promptArgs.Count > 0)
        {
            var prompt = string.Join(' ', promptArgs);
            Console.WriteLine($"User: {prompt}");

            var options = new GenerationOptions { MaxTokens = maxTokens };
            var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            Console.Write("Assistant: ");
            foreach (var text in session.GenerateAsync(prompt, options, cts.Token).ToBlockingEnumerable())
                Console.Write(text);
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("Interactive chat mode. Type 'quit' or Ctrl+C to exit.");
            Console.WriteLine("Type 'reset' to clear conversation history.");
            Console.WriteLine();

            while (true)
            {
                Console.Write("User: ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input)) continue;
                if (input.Equals("quit", StringComparison.OrdinalIgnoreCase)) break;
                if (input.Equals("reset", StringComparison.OrdinalIgnoreCase))
                {
                    session.Reset();
                    Console.WriteLine("History cleared.");
                    continue;
                }

                var options = new GenerationOptions { MaxTokens = maxTokens };
                var cts = new CancellationTokenSource();
                Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

                Console.Write("Assistant: ");
                foreach (var text in session.GenerateAsync(input, options, cts.Token).ToBlockingEnumerable())
                    Console.Write(text);
                Console.WriteLine();
                Console.WriteLine();
            }
        }
    }

    private static void RunRawMode(LoadedModel model, List<string> promptArgs)
    {
        var engine = new InferenceEngine(model);

        if (promptArgs.Count > 0)
        {
            var prompt = string.Join(' ', promptArgs);
            Console.Write("> ");
            Console.WriteLine(prompt);

            var tokens = PreparePromptTokens(model, prompt);
            Console.WriteLine($"  Encoded {tokens.Length} tokens");
            RunInference(engine, model, tokens, 64);
        }
        else
        {
            Console.WriteLine("Interactive mode (raw completion). Type 'quit' to exit.");
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