using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Llmdot.Inference;
using Llmdot.Loading;
using Llmdot.Tensors;

namespace Llmdot.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class MatMulBenchmarks : IDisposable
{
    private float[] _input = null!;
    private float[] _bF32 = null!;
    private byte[] _bQ40 = null!;
    private byte[] _bQ80 = null!;
    private float[] _result = null!;
    private CpuBackend _cpu = null!;

    [Params(2048)]
    public int HiddenSize { get; set; }

    [Params(32000)]
    public int OutputCols { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _input = new float[HiddenSize];
        _bF32 = new float[HiddenSize * OutputCols];
        _result = new float[OutputCols];

        for (var i = 0; i < HiddenSize; i++)
            _input[i] = rng.NextSingle() * 2 - 1;
        for (var i = 0; i < HiddenSize * OutputCols; i++)
            _bF32[i] = rng.NextSingle() * 2 - 1;

        var blocksPerRow = HiddenSize / 32;
        var q40BlockBytes = 18;
        var q80BlockBytes = 34;
        _bQ40 = new byte[OutputCols * blocksPerRow * q40BlockBytes];
        _bQ80 = new byte[OutputCols * blocksPerRow * q80BlockBytes];
        rng.NextBytes(_bQ40);
        rng.NextBytes(_bQ80);

        _cpu = new CpuBackend();
    }

    [Benchmark(Baseline = true)]
    public void CpuMatMulF32() =>
        _cpu.MatMulF32(_input, _bF32, _result, HiddenSize, OutputCols);

    [Benchmark]
    public void CpuMatMulQ4_0() =>
        _cpu.MatMul(_input, _bQ40, _result, GgmlType.Q4_0, HiddenSize, OutputCols);

    [Benchmark]
    public void CpuMatMulQ8_0() =>
        _cpu.MatMul(_input, _bQ80, _result, GgmlType.Q8_0, HiddenSize, OutputCols);

    public void Dispose()
    {
        _cpu.Dispose();
        GC.SuppressFinalize(this);
    }
}