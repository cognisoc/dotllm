using Llmdot.Inference;

namespace Llmdot.Cli.Commands;

internal static class InfoCommand
{
    public static int Run(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            CliOutput.WriteError($"Model file not found: {modelPath}");
            return 1;
        }

        using var stream = File.OpenRead(modelPath);
        using var model = LoadedModel.Load(stream);
        var config = model.Config;
        var caps = model.Capabilities;

        CliOutput.WriteHeader("Model Information");
        Console.WriteLine();

        CliOutput.WriteTable([
            ("Architecture:", config.Architecture),
            ("Template:", config.Template.ToString()),
            ("Hidden Size:", config.HiddenSize.ToString()),
            ("Layer Count:", config.LayerCount.ToString()),
            ("Context Length:", config.ContextLength.ToString()),
            ("Vocab Size:", config.VocabSize.ToString()),
            ("FFN Dim:", config.FfnDim.ToString()),
        ]);

        Console.WriteLine();
        CliOutput.WriteHeader("Attention");
        Console.WriteLine();

        CliOutput.WriteTable([
            ("Type:", config.AttentionType.ToString()),
            ("Head Count:", config.HeadCount.ToString()),
            ("Head Count KV:", config.HeadCountKv.ToString()),
            ("Head Dim:", config.HeadDim.ToString()),
            ("QKV Layout:", config.QkvLayout.ToString()),
        ]);

        Console.WriteLine();
        CliOutput.WriteHeader("Features");
        Console.WriteLine();

        CliOutput.WriteTable([
            ("Norm Type:", config.NormType.ToString()),
            ("FFN Type:", config.FfnType.ToString()),
            ("RoPE Freq Base:", config.RopeFreqBase.ToString("F1")),
            ("Tied Embeddings:", config.TiedEmbeddings.ToString()),
            ("Has Conv Layers:", caps.HasConvLayers.ToString()),
            ("Has MoE:", caps.HasMoE.ToString()),
            ("Expert Count:", config.ExpertCount.ToString()),
            ("Sliding Window:", config.SlidingWindow.ToString()),
        ]);

        if (model.ChatTemplate is not null)
        {
            Console.WriteLine();
            CliOutput.WriteHeader("Chat Template");
            Console.WriteLine();
            Console.WriteLine(model.ChatTemplate.RawTemplate.Length > 200
                ? model.ChatTemplate.RawTemplate[..200] + "..."
                : model.ChatTemplate.RawTemplate);
        }

        return 0;
    }
}
