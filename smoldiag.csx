using Llmdot.Inference;
using Llmdot.Models;
using Llmdot.Sampling;

using var stream = File.OpenRead(args[0]);
using var model = LoadedModel.Load(stream);

Console.WriteLine($"Architecture: {model.Config.Architecture}");
Console.WriteLine($"HiddenSize: {model.Config.HiddenSize}");
Console.WriteLine($"HeadCount: {model.Config.HeadCount}, HeadCountKv: {model.Config.HeadCountKv}");
Console.WriteLine($"HeadDim: {model.Config.HeadDim}, QDim: {model.Config.QDim}, KvDim: {model.Config.KvDim}");
Console.WriteLine($"TiedEmbeddings: {model.Config.TiedEmbeddings}");
Console.WriteLine($"VocabSize: {model.Config.VocabSize}");
Console.WriteLine($"BOS: {model.Config.BosTokenId}, EOS: {model.Config.EosTokenId}");

var engine = new InferenceEngine(model);
var promptTokens = model.Tokenizer.Encode("What is 2+2?");
if (model.Config.BosTokenId > 0)
    promptTokens = [model.Config.BosTokenId, .. promptTokens];

Console.WriteLine($"Prompt tokens: [{string.Join(", ", promptTokens)}]");
Console.WriteLine($"Prompt decoded: {model.Tokenizer.Decode(promptTokens)}");

var options = new GenerationOptions
{
    MaxTokens = 20,
    Sampling = new SamplingOptions { Temperature = 0f }
};

Console.Write("Output: ");
await foreach (var tokenId in engine.Generate(promptTokens, options))
{
    Console.Write(model.Tokenizer.Decode([tokenId]));
    if (tokenId == model.Config.EosTokenId) break;
}
Console.WriteLine();