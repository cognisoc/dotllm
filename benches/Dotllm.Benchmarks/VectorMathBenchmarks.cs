using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Dotllm.Tensors.Numeric;

namespace Dotllm.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90)]
[MemoryDiagnoser]
public class VectorMathBenchmarks
{
    private float[] _input = null!;
    private float[] _output = null!;
    private float[] _weights = null!;
    private float[] _bias = null!;

    [Params(512, 2048, 4096)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _input = new float[Size];
        _output = new float[Size];
        _weights = new float[Size];
        _bias = new float[Size];

        for (var i = 0; i < Size; i++)
        {
            _input[i] = rng.NextSingle() * 2 - 1;
            _weights[i] = rng.NextSingle() * 2 - 1;
            _bias[i] = rng.NextSingle() * 0.1f;
        }
    }

    [Benchmark]
    public void RmsNorm() =>
        VectorMath.RmsNorm(_input, _weights, _output, 1e-5f);

    [Benchmark]
    public void LayerNorm() =>
        VectorMath.LayerNorm(_input, _weights, _bias, _output, 1e-5f);

    [Benchmark]
    public void Softmax()
    {
        _input.AsSpan().CopyTo(_output);
        VectorMath.Softmax(_output);
    }

    [Benchmark]
    public void Add() =>
        VectorMath.Add(_input, _weights, _output);

    [Benchmark]
    public void Scale() =>
        VectorMath.Scale(_input, 0.125f, _output);

    [Benchmark]
    public void Mul() =>
        VectorMath.Mul(_input, _weights, _output);

    [Benchmark]
    public void Silu() =>
        VectorMath.Silu(_input, _output);

    [Benchmark]
    public void Gelu() =>
        VectorMath.Gelu(_input, _output);
}