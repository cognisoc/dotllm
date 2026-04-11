namespace Dotllm.Inference;

internal sealed class KvCache
{
    private readonly float[] _keyBuffer;
    private readonly float[] _valueBuffer;
    private readonly int _maxSeqLen;
    private readonly int _kvDim;
    private readonly int _layerCount;
    private int _currentPos;

    public int CurrentPosition => _currentPos;

    public KvCache(int layerCount, int kvDim, int maxSeqLen)
    {
        _layerCount = layerCount;
        _kvDim = kvDim;
        _maxSeqLen = maxSeqLen;
        _keyBuffer = new float[layerCount * maxSeqLen * kvDim];
        _valueBuffer = new float[layerCount * maxSeqLen * kvDim];
    }

    public Span<float> GetKeys(int layer) =>
        _keyBuffer.AsSpan(layer * _maxSeqLen * _kvDim, _currentPos * _kvDim);

    public Span<float> GetValues(int layer) =>
        _valueBuffer.AsSpan(layer * _maxSeqLen * _kvDim, _currentPos * _kvDim);

    public Span<float> GetKeySlot(int layer, int position)
    {
        var offset = layer * _maxSeqLen * _kvDim + position * _kvDim;
        return _keyBuffer.AsSpan(offset, _kvDim);
    }

    public Span<float> GetValueSlot(int layer, int position)
    {
        var offset = layer * _maxSeqLen * _kvDim + position * _kvDim;
        return _valueBuffer.AsSpan(offset, _kvDim);
    }

    public void Advance(int count = 1) => _currentPos += count;

    public void Reset() => _currentPos = 0;
}