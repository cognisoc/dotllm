namespace Llmdot.Inference;

internal sealed class ConvStateCache
{
    private readonly float[] _buffer;
    private readonly float[] _scratchBuf;
    private readonly int[] _head;
    private readonly int _hiddenSize;
    private readonly int _historyLen;
    private readonly int _layerCount;
    private readonly int _kernelSize;

    public ConvStateCache(int layerCount, int hiddenSize, int kernelSize)
    {
        _layerCount = layerCount;
        _hiddenSize = hiddenSize;
        _historyLen = kernelSize - 1;
        _kernelSize = kernelSize;
        _buffer = new float[layerCount * _historyLen * hiddenSize];
        _scratchBuf = new float[kernelSize * hiddenSize];
        _head = new int[layerCount];
    }

    public void Store(int layer, ReadOnlySpan<float> channelData)
    {
        if (_historyLen == 0) return;
        var pos = _head[layer];
        var offset = (layer * _historyLen + pos) * _hiddenSize;
        channelData.Slice(0, _hiddenSize).CopyTo(_buffer.AsSpan(offset, _hiddenSize));
        _head[layer] = (pos + 1) % _historyLen;
    }

    public Span<float> BuildInput(int layer, ReadOnlySpan<float> current, int position)
    {
        var totalSize = _kernelSize;
        var result = _scratchBuf.AsSpan(0, totalSize * _hiddenSize);

        for (var c = 0; c < _hiddenSize; c++)
        {
            for (var k = 0; k < _historyLen; k++)
            {
                var requiredPos = position - _historyLen + k;
                if (requiredPos < 0)
                {
                    result[c * totalSize + k] = 0f;
                }
                else
                {
                    var slot = requiredPos % _historyLen;
                    var srcOff = (layer * _historyLen + slot) * _hiddenSize + c;
                    result[c * totalSize + k] = _buffer[srcOff];
                }
            }

            result[c * totalSize + _historyLen] = current[c];
        }

        return result;
    }

    public void Reset()
    {
        Array.Clear(_buffer);
        Array.Clear(_head);
    }
}