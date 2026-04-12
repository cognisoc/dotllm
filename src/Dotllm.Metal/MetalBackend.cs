using System.Runtime.InteropServices;
using Dotllm.Inference;
using Dotllm.Loading;
using Dotllm.Metal.Native;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Metal;

public sealed class MetalBackend : IComputeBackend
{
    private readonly MetalInterop? _metal;
    private readonly Dictionary<string, nint> _pipelines = [];
    private readonly List<nint> _scratchBuffers = [];
    private readonly Dictionary<string, nint> _persistentBuffers = [];
    private bool _disposed;
    private bool _gpuAvailable;

    public MetalBackend()
    {
        try
        {
            _metal = new MetalInterop();
            CompileAllShaders();
            _gpuAvailable = true;
        }
        catch
        {
            _gpuAvailable = false;
        }
    }

    public static bool IsAvailable => MetalRuntime.IsAvailable;

    private void CompileAllShaders()
    {
        if (_metal == null) return;

        CompileKernel("rms_norm", MetalShaders.RmsNorm);
        CompileKernel("layer_norm", MetalShaders.LayerNorm);
        CompileKernel("softmax", MetalShaders.Softmax);
        CompileKernel("add", MetalShaders.Add);
        CompileKernel("add_inplace", MetalShaders.AddInPlace);
        CompileKernel("scale", MetalShaders.Scale);
        CompileKernel("scale_inplace", MetalShaders.ScaleInPlace);
        CompileKernel("mul", MetalShaders.Mul);
        CompileKernel("silu", MetalShaders.Silu);
        CompileKernel("silu_inplace", MetalShaders.SiluInPlace);
        CompileKernel("gelu", MetalShaders.Gelu);
        CompileKernel("matmul_f32", MetalShaders.MatMulF32);
    }

    private void CompileKernel(string name, string source)
    {
        var pipeline = _metal!.CompileShader(source, name);
        _pipelines[name] = pipeline;
    }

    private nint AllocScratch(ulong sizeBytes)
    {
        var buffer = _metal!.CreateBuffer(sizeBytes);
        _scratchBuffers.Add(buffer);
        return buffer;
    }

    private nint UploadScratch(ReadOnlySpan<float> data)
    {
        var buffer = _metal!.CreateBufferFromFloats(data);
        _scratchBuffers.Add(buffer);
        return buffer;
    }

    private nint UploadScratchBytes(ReadOnlySpan<byte> data)
    {
        var buffer = _metal!.CreateBufferFromData(data);
        _scratchBuffers.Add(buffer);
        return buffer;
    }

    public nint UploadPersistentFloats(string key, ReadOnlySpan<float> data)
    {
        if (_persistentBuffers.TryGetValue(key, out var existing))
            return existing;

        var buffer = _metal!.CreateBufferFromFloats(data);
        _persistentBuffers[key] = buffer;
        return buffer;
    }

    public nint UploadPersistentBytes(string key, ReadOnlySpan<byte> data)
    {
        if (_persistentBuffers.TryGetValue(key, out var existing))
            return existing;

        var buffer = _metal!.CreateBufferFromData(data);
        _persistentBuffers[key] = buffer;
        return buffer;
    }

    public nint AllocatePersistent(string key, ulong sizeBytes)
    {
        if (_persistentBuffers.TryGetValue(key, out var existing))
            return existing;

        var buffer = _metal!.CreateBuffer(sizeBytes);
        _persistentBuffers[key] = buffer;
        return buffer;
    }

    public bool HasPersistent(string key) => _persistentBuffers.ContainsKey(key);

    private void ReleaseScratch()
    {
        foreach (var handle in _scratchBuffers)
            MetalInterop.DestroyBuffer(handle);
        _scratchBuffers.Clear();
    }

    private sealed class ComputeScope : IDisposable
    {
        public nint CommandBuffer { get; }
        public nint Encoder { get; }

        public ComputeScope(MetalInterop metal)
        {
            CommandBuffer = metal.CreateCommandBuffer();
            Encoder = MetalInterop.CreateComputeEncoder(CommandBuffer);
        }

        public void Dispose()
        {
            MetalInterop.EndEncoding(Encoder);
            MetalInterop.Commit(CommandBuffer);
            MetalInterop.WaitUntilCompleted(CommandBuffer);
        }
    }

    private void Run1D(string kernel, Span<(nint buf, uint idx)> buffers, uint count, uint threadsPerGroup = 256)
    {
        if (!_gpuAvailable || _metal == null) return;

        using var scope = new ComputeScope(_metal);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines[kernel]);
        foreach (var (buf, idx) in buffers)
            MetalInterop.SetBuffer(scope.Encoder, buf, 0, idx);
        MetalInterop.Dispatch1D(scope.Encoder, count, threadsPerGroup);
    }

    private void GpuRmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon)
    {
        var m = _metal!;
        var inputBuf = UploadScratch(input);
        var weightsBuf = UploadScratch(weights);
        var outputBuf = AllocScratch((ulong)output.Length * 4);
        var hiddenSize = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["rms_norm"]);
        MetalInterop.SetBuffer(scope.Encoder, inputBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, weightsBuf, 0, 1);
        MetalInterop.SetBuffer(scope.Encoder, outputBuf, 0, 2);
        MetalInterop.SetBytesFloat(scope.Encoder, epsilon, 3);
        MetalInterop.SetBytesUint(scope.Encoder, hiddenSize, 4);
        MetalInterop.Dispatch1D(scope.Encoder, hiddenSize, 256);

        m.CopyFromBuffer(outputBuf, output);
        ReleaseScratch();
    }

    private void GpuLayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon)
    {
        var m = _metal!;
        var inputBuf = UploadScratch(input);
        var weightsBuf = UploadScratch(weights);
        var biasBuf = UploadScratch(bias);
        var outputBuf = AllocScratch((ulong)output.Length * 4);
        var hiddenSize = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["layer_norm"]);
        MetalInterop.SetBuffer(scope.Encoder, inputBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, weightsBuf, 0, 1);
        MetalInterop.SetBuffer(scope.Encoder, biasBuf, 0, 2);
        MetalInterop.SetBuffer(scope.Encoder, outputBuf, 0, 3);
        MetalInterop.SetBytesFloat(scope.Encoder, epsilon, 4);
        MetalInterop.SetBytesUint(scope.Encoder, hiddenSize, 5);
        MetalInterop.Dispatch1D(scope.Encoder, hiddenSize, 256);

        m.CopyFromBuffer(outputBuf, output);
        ReleaseScratch();
    }

    private void GpuSoftmax(Span<float> input, float softcapVal)
    {
        var m = _metal!;
        var inputBuf = AllocScratch((ulong)input.Length * 4);
        m.CopyToBuffer(inputBuf, input);

        var len = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["softmax"]);
        MetalInterop.SetBuffer(scope.Encoder, inputBuf, 0, 0);
        MetalInterop.SetBytesFloat(scope.Encoder, softcapVal, 1);
        MetalInterop.SetBytesUint(scope.Encoder, len, 2);
        MetalInterop.Dispatch1D(scope.Encoder, len, 256);

        m.CopyFromBuffer(inputBuf, input);
        ReleaseScratch();
    }

    private void GpuElementwise(string kernel, ReadOnlySpan<float> input, Span<float> result)
    {
        var m = _metal!;
        var inputBuf = UploadScratch(input);
        var resultBuf = AllocScratch((ulong)result.Length * 4);
        var count = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines[kernel]);
        MetalInterop.SetBuffer(scope.Encoder, inputBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, resultBuf, 0, 1);
        MetalInterop.SetBytesUint(scope.Encoder, count, 2);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(resultBuf, result);
        ReleaseScratch();
    }

    private void GpuElementwiseInPlace(string kernel, Span<float> input)
    {
        var m = _metal!;
        var buf = AllocScratch((ulong)input.Length * 4);
        m.CopyToBuffer(buf, input);
        var count = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines[kernel]);
        MetalInterop.SetBuffer(scope.Encoder, buf, 0, 0);
        MetalInterop.SetBytesUint(scope.Encoder, count, 1);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(buf, input);
        ReleaseScratch();
    }

    private void GpuAddInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        var m = _metal!;
        var aBuf = AllocScratch((ulong)a.Length * 4);
        var bBuf = UploadScratch(b);
        m.CopyToBuffer(aBuf, a);
        var count = (uint)a.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["add_inplace"]);
        MetalInterop.SetBuffer(scope.Encoder, aBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, bBuf, 0, 1);
        MetalInterop.SetBytesUint(scope.Encoder, count, 2);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(aBuf, a);
        ReleaseScratch();
    }

    private void GpuAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        var m = _metal!;
        var aBuf = UploadScratch(a);
        var bBuf = UploadScratch(b);
        var resultBuf = AllocScratch((ulong)result.Length * 4);
        var count = (uint)a.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["add"]);
        MetalInterop.SetBuffer(scope.Encoder, aBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, bBuf, 0, 1);
        MetalInterop.SetBuffer(scope.Encoder, resultBuf, 0, 2);
        MetalInterop.SetBytesUint(scope.Encoder, count, 3);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(resultBuf, result);
        ReleaseScratch();
    }

    private void GpuScale(ReadOnlySpan<float> input, float scale, Span<float> result)
    {
        var m = _metal!;
        var inputBuf = UploadScratch(input);
        var resultBuf = AllocScratch((ulong)result.Length * 4);
        var count = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["scale"]);
        MetalInterop.SetBuffer(scope.Encoder, inputBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, resultBuf, 0, 1);
        MetalInterop.SetBytesFloat(scope.Encoder, scale, 2);
        MetalInterop.SetBytesUint(scope.Encoder, count, 3);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(resultBuf, result);
        ReleaseScratch();
    }

    private void GpuScaleInPlace(Span<float> input, float scale)
    {
        var m = _metal!;
        var buf = AllocScratch((ulong)input.Length * 4);
        m.CopyToBuffer(buf, input);
        var count = (uint)input.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["scale_inplace"]);
        MetalInterop.SetBuffer(scope.Encoder, buf, 0, 0);
        MetalInterop.SetBytesFloat(scope.Encoder, scale, 1);
        MetalInterop.SetBytesUint(scope.Encoder, count, 2);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(buf, input);
        ReleaseScratch();
    }

    private void GpuMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        var m = _metal!;
        var aBuf = UploadScratch(a);
        var bBuf = UploadScratch(b);
        var resultBuf = AllocScratch((ulong)result.Length * 4);
        var count = (uint)a.Length;

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["mul"]);
        MetalInterop.SetBuffer(scope.Encoder, aBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, bBuf, 0, 1);
        MetalInterop.SetBuffer(scope.Encoder, resultBuf, 0, 2);
        MetalInterop.SetBytesUint(scope.Encoder, count, 3);
        MetalInterop.Dispatch1D(scope.Encoder, count, 256);

        m.CopyFromBuffer(resultBuf, result);
        ReleaseScratch();
    }

    private void GpuMatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        var m = _metal!;
        var aBuf = UploadScratch(a);
        var bBuf = UploadScratch(b);
        var resultBuf = AllocScratch((ulong)result.Length * 4);

        using var scope = new ComputeScope(m);
        MetalInterop.SetPipeline(scope.Encoder, _pipelines["matmul_f32"]);
        MetalInterop.SetBuffer(scope.Encoder, aBuf, 0, 0);
        MetalInterop.SetBuffer(scope.Encoder, bBuf, 0, 1);
        MetalInterop.SetBuffer(scope.Encoder, resultBuf, 0, 2);
        MetalInterop.SetBytesUint(scope.Encoder, (uint)aCols, 3);
        MetalInterop.SetBytesUint(scope.Encoder, (uint)bCols, 4);
        MetalInterop.Dispatch1D(scope.Encoder, (uint)bCols, 256);

        m.CopyFromBuffer(resultBuf, result);
        ReleaseScratch();
    }

    public void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, float epsilon)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuRmsNorm(input, weights, output, epsilon);
        else
            VectorMath.RmsNorm(input, weights, output, epsilon);
    }

    public void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, ReadOnlySpan<float> bias, Span<float> output, float epsilon)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuLayerNorm(input, weights, bias, output, epsilon);
        else
            VectorMath.LayerNorm(input, weights, bias, output, epsilon);
    }

    public void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, float freqBase, int rotaryDim)
        => Tensors.TensorOps.ApplyRoPE(query, key, headDim, position, freqBase, rotaryDim);

    public void ApplyRoPE(Span<float> query, Span<float> key, int headDim, int position, ReadOnlySpan<float> freqTable)
        => Tensors.TensorOps.ApplyRoPE(query, key, headDim, position, freqTable);

    public void Softmax(Span<float> input, float? softcap = null)
    {
        if (_gpuAvailable && _metal != null)
            GpuSoftmax(input, softcap ?? 0f);
        else
            VectorMath.Softmax(input, softcap);
    }

    public void Silu(ReadOnlySpan<float> input, Span<float> result)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuElementwise("silu", input, result);
        else
            VectorMath.Silu(input, result);
    }

    public void SiluInPlace(Span<float> input)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuElementwiseInPlace("silu_inplace", input);
        else
            VectorMath.SiluInPlace(input);
    }

    public void Gelu(ReadOnlySpan<float> input, Span<float> result)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuElementwise("gelu", input, result);
        else
            VectorMath.Gelu(input, result);
    }

    public float GeluScalar(float x) => VectorMath.Gelu(x);

    public void Add(Span<float> a, ReadOnlySpan<float> b)
    {
        if (_gpuAvailable && _metal != null && a.Length >= 256)
            GpuAddInPlace(a, b);
        else
            VectorMath.Add(a, b);
    }

    public void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (_gpuAvailable && _metal != null && a.Length >= 256)
            GpuAdd(a, b, result);
        else
            VectorMath.Add(a, b, result);
    }

    public void Scale(ReadOnlySpan<float> input, float scale, Span<float> result)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuScale(input, scale, result);
        else
            VectorMath.Scale(input, scale, result);
    }

    public void Scale(Span<float> input, float scale)
    {
        if (_gpuAvailable && _metal != null && input.Length >= 256)
            GpuScaleInPlace(input, scale);
        else
            VectorMath.Scale(input, scale);
    }

    public void Mul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (_gpuAvailable && _metal != null && a.Length >= 256)
            GpuMul(a, b, result);
        else
            VectorMath.Mul(a, b, result);
    }

    public void Softcap(Span<float> input, float cap) => VectorMath.Softcap(input, cap);

    public void Conv1D(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output, int kernelSize, int inputDim)
        => Tensors.TensorOps.Conv1D(input, weights, output, kernelSize, inputDim);

    public void DequantizeToFloat(ReadOnlySpan<byte> src, Span<float> dst, GgmlType type, int numRows, int rowElements)
        => Tensors.TensorOps.DequantizeToFloat(src, dst, type, numRows, rowElements);

    public int ArgMax(ReadOnlySpan<float> input) => VectorMath.ArgMax(input);

    public void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<byte> b, Span<float> result, GgmlType bType, int aCols, int bCols)
        => Tensors.TensorOps.MatMul(a, b, result, bType, aCols, bCols);

    public void MatMulF32(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result, int aCols, int bCols)
    {
        if (_gpuAvailable && _metal != null && bCols >= 256 && aCols >= 256)
            GpuMatMulF32(a, b, result, aCols, bCols);
        else
            Tensors.TensorOps.MatMulF32(a, b, result, aCols, bCols);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var handle in _scratchBuffers)
            MetalInterop.DestroyBuffer(handle);
        _scratchBuffers.Clear();

        foreach (var (_, handle) in _persistentBuffers)
            MetalInterop.DestroyBuffer(handle);
        _persistentBuffers.Clear();

        foreach (var (_, handle) in _pipelines)
            MetalInterop.DestroyPipeline(handle);
        _pipelines.Clear();

        _metal?.Dispose();
    }
}