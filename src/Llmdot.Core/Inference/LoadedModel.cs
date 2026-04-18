using System.Runtime.InteropServices;
using Llmdot.Loading;
using Llmdot.Models;
using Llmdot.Tensors;
using Llmdot.Tokenization;

namespace Llmdot.Inference;

/// <summary>
/// Represents a GGUF model loaded into memory, ready for inference.
/// The stream passed to <see cref="Load"/> is owned by this instance and will be
/// disposed when <see cref="Dispose"/> is called.
/// </summary>
public sealed class LoadedModel : IDisposable
{
    private readonly Stream _stream;
    private readonly GgufModel _ggufModel;
    private readonly Dictionary<string, Tensor> _tensors;
    private readonly Dictionary<string, float[]> _dequantizedCache;
    private bool _disposed;

    public TransformerConfig Config { get; }
    internal TensorNameResolver TensorNames { get; }
    public BpeTokenizer Tokenizer { get; }
    public ChatTemplate? ChatTemplate { get; }
    public ModelCapabilities Capabilities { get; }

    private LoadedModel(Stream stream, GgufModel ggufModel, TransformerConfig config, TensorNameResolver tensorNames, Dictionary<string, Tensor> tensors, BpeTokenizer tokenizer, ChatTemplate? chatTemplate)
    {
        _stream = stream;
        _ggufModel = ggufModel;
        Config = config;
        TensorNames = tensorNames;
        _tensors = tensors;
        _dequantizedCache = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
        Tokenizer = tokenizer;
        ChatTemplate = chatTemplate;
        Capabilities = ModelCapabilities.FromConfig(config);
    }

    /// <summary>
    /// Loads a GGUF model from the provided stream. The stream is owned by the
    /// returned <see cref="LoadedModel"/> and will be disposed when the model is disposed.
    /// </summary>
    public static LoadedModel Load(Stream stream)
    {
        var ggufModel = GgufReader.Read(stream);
        var config = ArchitectureResolver.Resolve(ggufModel);
        var tensorNames = new TensorNameResolver(ggufModel.TensorInfos);
        var tensors = LoadTensors(stream, ggufModel);
        var tokenizer = BpeTokenizer.FromGguf(ggufModel.Metadata);
        var chatTemplate = ChatTemplate.FromGguf(ggufModel.Metadata, tokenizer);

        return new LoadedModel(stream, ggufModel, config, tensorNames, tensors, tokenizer, chatTemplate);
    }

    internal Tensor GetTensor(string name) =>
        _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' not found.");

    internal bool TryGetTensor(string name, out Tensor? tensor) =>
        _tensors.TryGetValue(name, out tensor);

    internal float[] GetDequantizedWeights(string name)
    {
        if (_dequantizedCache.TryGetValue(name, out var cached))
            return cached;

        var tensor = GetTensor(name);
        var size = (int)tensor.ElementCount;
        var result = new float[size];
        var byteCount = (int)TensorSize.ByteCount(tensor.ElementType, (ulong)size);
        TensorOps.DequantizeToFloat(tensor.Data.Span[..byteCount], result, tensor.ElementType, tensor.RowCount > 0 ? tensor.RowCount : 1, size / Math.Max(tensor.RowCount, 1));
        _dequantizedCache[name] = result;
        return result;
    }

    private static Dictionary<string, Tensor> LoadTensors(Stream stream, GgufModel model)
    {
        var tensors = new Dictionary<string, Tensor>(model.TensorInfos.Count, StringComparer.OrdinalIgnoreCase);

        foreach (var info in model.TensorInfos)
        {
            var byteCount = (long)TensorSize.ByteCount(info.Type, info.ElementCount);
            if (byteCount == 0) continue;

            var dataOffset = (long)model.TensorDataOffset + (long)info.Offset;
            if (dataOffset + byteCount > stream.Length)
                continue;

            stream.Position = dataOffset;
            var data = new byte[byteCount];
            var read = stream.Read(data, 0, (int)byteCount);
            if (read != (int)byteCount)
                continue;

            tensors[info.Name] = new Tensor
            {
                Name = info.Name,
                Dimensions = info.Dimensions,
                ElementType = info.Type,
                Data = data,
                ElementCount = info.ElementCount,
            };
        }

        return tensors;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _stream.Dispose();
    }

    public Loading.ModelValidationResult Validate()
    {
        return Loading.ModelValidator.Validate(_ggufModel, Config);
    }
}