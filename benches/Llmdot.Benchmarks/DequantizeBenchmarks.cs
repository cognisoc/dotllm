using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Llmdot.Loading;
using Llmdot.Tensors;
using Llmdot.Tensors.Dequantize;
using Llmdot.Tensors.Numeric;

namespace Llmdot.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class DequantizeBenchmarks
{
    private byte[] _q40Data = null!;
    private byte[] _q80Data = null!;
    private float[] _output = null!;

    [Params(512, 2048)]
    public int RowElements { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        var blockSizeQ40 = 18;
        var blockSizeQ80 = 34;
        var blocksPerRow = RowElements / 32;

        _q40Data = new byte[4 * blocksPerRow * blockSizeQ40];
        _q80Data = new byte[4 * blocksPerRow * blockSizeQ80];
        _output = new float[RowElements * 4];

        rng.NextBytes(_q40Data);
        rng.NextBytes(_q80Data);
    }

    [Benchmark]
    public void BenchmarkQ4_0() =>
        DequantizeQ4_0.Dequantize(_q40Data, _output, 4, RowElements);

    [Benchmark]
    public void BenchmarkQ8_0() =>
        DequantizeQ8_0.Dequantize(_q80Data, _output, 4, RowElements);
}