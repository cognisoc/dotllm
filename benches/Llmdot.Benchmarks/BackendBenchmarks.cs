using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Llmdot.Inference;
using Llmdot.Metal;
using Llmdot.Tensors.Numeric;

namespace Llmdot.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
[RankColumn]
public class BackendBenchmarks : IDisposable
{
    private float[] _input = null!;
    private float[] _output = null!;
    private float[] _weights = null!;
    private CpuBackend _cpu = null!;
    private MetalBackend? _metal;

    [Params(2048)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _input = new float[Size];
        _output = new float[Size];
        _weights = new float[Size];

        for (var i = 0; i < Size; i++)
            _input[i] = rng.NextSingle() * 2 - 1;
        for (var i = 0; i < Size; i++)
            _weights[i] = rng.NextSingle() * 2 - 1;

        _cpu = new CpuBackend();

        if (MetalBackend.IsAvailable)
        {
            try { _metal = new MetalBackend(); }
            catch { _metal = null; }
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cpu.Dispose();
        _metal?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public void CpuRmsNorm() =>
        VectorMath.RmsNorm(_input, _weights, _output, 1e-5f);

    [Benchmark]
    public void MetalRmsNorm()
    {
        if (_metal == null) return;
        _metal.RmsNorm(_input, _weights, _output, 1e-5f);
    }

    [Benchmark]
    public void CpuSoftmax()
    {
        _input.AsSpan().CopyTo(_output);
        VectorMath.Softmax(_output);
    }

    [Benchmark]
    public void MetalSoftmax()
    {
        if (_metal == null) return;
        _input.AsSpan().CopyTo(_output);
        _metal.Softmax(_output);
    }

    [Benchmark]
    public void CpuSilu() =>
        VectorMath.Silu(_input, _output);

    [Benchmark]
    public void MetalSilu()
    {
        if (_metal == null) return;
        _metal.Silu(_input, _output);
    }

    [Benchmark]
    public void CpuAdd() =>
        VectorMath.Add(_input, _weights, _output);

    [Benchmark]
    public void MetalAdd()
    {
        if (_metal == null) return;
        _metal.Add(_input, _weights, _output);
    }

    [Benchmark]
    public void CpuMul() =>
        VectorMath.Mul(_input, _weights, _output);

    [Benchmark]
    public void MetalMul()
    {
        if (_metal == null) return;
        _metal.Mul(_input, _weights, _output);
    }

    public void Dispose()
    {
        _cpu.Dispose();
        _metal?.Dispose();
        GC.SuppressFinalize(this);
    }
}