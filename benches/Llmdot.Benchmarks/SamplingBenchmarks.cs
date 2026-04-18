using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace Llmdot.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class SamplingBenchmarks
{
    private float[] _logits = null!;
    private readonly Llmdot.Sampling.SamplingOptions _options = new() { Temperature = 0.8f, TopK = 40, TopP = 0.95f };
    private readonly Llmdot.Sampling.SamplingOptions _greedy = new() { Temperature = 0f };

    [Params(32000, 128000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _logits = new float[VocabSize];
        for (var i = 0; i < VocabSize; i++)
            _logits[i] = rng.NextSingle() * 20 - 10;
    }

    [Benchmark]
    public int ArgMax() =>
        Llmdot.Tensors.Numeric.VectorMath.ArgMax(_logits);
}