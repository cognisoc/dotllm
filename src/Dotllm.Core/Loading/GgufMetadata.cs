using System.Collections.ObjectModel;

namespace Dotllm.Loading;

internal sealed class GgufMetadata
{
    private readonly Dictionary<string, GgufMetadataValue> _values;

    public GgufMetadata(Dictionary<string, GgufMetadataValue> values)
    {
        _values = values;
        Values = new ReadOnlyDictionary<string, GgufMetadataValue>(values);
    }

    public IReadOnlyDictionary<string, GgufMetadataValue> Values { get; }

    public bool TryGetValue(string key, out GgufMetadataValue? value) =>
        _values.TryGetValue(key, out value);

    public T? GetOrDefault<T>(string key, T? defaultValue)
    {
        if (_values.TryGetValue(key, out var v) && v.Value is T t)
            return t;
        return defaultValue;
    }
}

internal sealed class GgufMetadataValue
{
    public GgufValueType Type { get; init; }
    public object Value { get; init; } = default!;
}