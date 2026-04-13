namespace Dotllm.Inference;

internal sealed class ConvStateCache
{
    private readonly float[] _buffer;
    private readonly float[] _scratchBuf;
    private readonly int[] _head;
    private readonly int _hiddenSize;
    private readonly int _historyLen;
    private readonly int _layerCount;

    public ConvStateCache(int layerCount, int hiddenSize, int kernelSize)
    {
        _layerCount = layerCount;
        _hiddenSize = hiddenSize;
        _historyLen = kernelSize - 1;
        _buffer = new float[layerCount * _historyLen * hiddenSize];
        _scratchBuf = new float[kernelSize * hiddenSize];
        _head = new int[layerCount];
    }

    public void Store(int layer, ReadOnlySpan<float> hiddenState)
    {
        var pos = _head[layer];
        var offset = (layer * _historyLen + pos) * _hiddenSize;
        hiddenState.CopyTo(_buffer.AsSpan(offset, _hiddenSize));
        _head[layer] = _historyLen > 0 ? (pos + 1) % _historyLen : 0;
    }

    public Span<float> BuildInput(int layer, ReadOnlySpan<float> current, int position)
    {
        var historySize = Math.Min(Math.Max(position, 0), _historyLen);
        var totalSize = 1 + historySize;
        var result = _scratchBuf.AsSpan(0, totalSize * _hiddenSize);

        current.CopyTo(result.Slice(0, _hiddenSize));

        for (var k = 0; k < historySize; k++)
        {
            var age = k + 1;
            var ringIdx = ((_head[layer] - age + _historyLen) % _historyLen);
            var srcOff = (layer * _historyLen + ringIdx) * _hiddenSize;
            var dstOff = (k + 1) * _hiddenSize;
            _buffer.AsSpan(srcOff, _hiddenSize).CopyTo(result.Slice(dstOff, _hiddenSize));
        }

        return result;
    }

    public void Reset()
    {
        Array.Clear(_buffer);
        Array.Clear(_head);
    }
}