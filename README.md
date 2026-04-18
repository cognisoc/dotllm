# llmdot

**Run local GGUF language models from .NET — one package, one format, one programming model.**

[![.NET](https://img.shields.io/badge/.NET-8.0%20%7C%209.0%20%7C%2010.0-512BD4?logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![NuGet](https://img.shields.io/badge/nuget-preview-004880?logo=nuget&logoColor=white)](#)
[![Build](https://img.shields.io/badge/build-in%20development-orange)](#)
[![GGUF](https://img.shields.io/badge/format-GGUF-8A2BE2)](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
[![AOT](https://img.shields.io/badge/NativeAOT-friendly-success)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![Status](https://img.shields.io/badge/status-pre--alpha-red)](doc/roadmap.md)

*A .NET-native local inference runtime for GGUF language models. CPU-first. Managed-by-default. Idiomatic. Trimming- and NativeAOT-friendly.*

[Vision](doc/vision.md) • [Architecture](doc/architecture.md) • [Roadmap](doc/roadmap.md) • [Platform Strategy](doc/platform-strategy.md) • [Model Reference](doc/model-architectures.md)

---

## What is llmdot

`llmdot` is a native .NET runtime for local language model inference built around the **GGUF** model format. It executes major decoder-only transformer and hybrid architectures in the **1–8B parameter range** — including multimodal variants — through architecture-agnostic execution templates resolved from GGUF metadata at load time.

The project is designed around a single opinionated goal: **make local LLM execution in .NET as simple as adding a NuGet package, loading a GGUF file, and streaming tokens**. The default path is pure managed code with zero native runtime dependencies, focused on CPU-first execution. Optional packages provide GPU acceleration through thin backend adapters.

```csharp
using Llmdot;

await using var model = await LlmModel.LoadAsync("phi-3-mini-q4_k_m.gguf");
await using var session = model.CreateChatSession();

await foreach (var token in session.StreamAsync("Explain GGUF in one paragraph."))
    Console.Write(token);
```

> The code sample above reflects the target API shape. See [roadmap](doc/roadmap.md) for the current state of the implementation.

---

## Why llmdot

The .NET inference landscape today forces developers into one of two uncomfortable tradeoffs:

| Option | Strength | Tradeoff |
|---|---|---|
| **`llama.cpp` bindings** | Broad model support | Native binaries, per-platform packaging, upstream integration debt |
| **ONNX-based stacks** | Strong hardware acceleration | Model conversion, large native dependencies, toolchain friction |

`llmdot` takes a third position:

- **GGUF-native** execution with no conversion pipeline
- **Pure managed core** — trimming-friendly, NativeAOT-friendly, single-file publish-friendly
- **Idiomatic .NET APIs** built for `IAsyncEnumerable<T>` streaming, DI, and `Microsoft.Extensions.Hosting`
- **Config-driven architectures** — new model families plug into existing execution templates with zero engine code
- **Focus on the common case** — small-to-mid quantized models where developer experience beats peak throughput

---

## Who llmdot is for

**.NET developers** who want to ship local, private, or offline AI features without fighting the inference stack.

**Software architects** evaluating local LLM runtimes for desktop, edge, and server workloads where packaging simplicity, deployment predictability, and platform portability matter as much as raw throughput.

**Teams building on `Microsoft.Extensions.AI`** who need an `IChatClient`-compatible backend that runs fully in-process, with no sidecar services and no native toolchain.

If you have ever thought *"I just want to load a GGUF file in my ASP.NET Core app and stream tokens"* — llmdot is built for you.

---

## Design Principles

1. **Zero native dependencies in the core path.** The default install is pure managed .NET. Native acceleration is always additive.
2. **GGUF is the ingestion format.** No ONNX conversion. No proprietary packaging. Community models work out of the box.
3. **Architecture support is declarative, not hard-coded.** New model families are resolved through `TransformerConfig` from GGUF metadata.
4. **Model compatibility is decoupled from hardware backend.** CPU, Vulkan, or Metal — same model, same code.
5. **Optimize for the common case.** 1–8B quantized models on consumer hardware. Small enough to fit, big enough to matter.
6. **Incremental acceleration.** Backends offload individual operations, not entire graphs. No all-or-nothing rewrites.

---

## Supported Architectures

All supported architectures collapse into **four execution templates**. Within each template, all variation is expressed through configuration — no per-model code paths.

| Template | Architectures | Example Models |
|---|---|---|
| **LLaMA-like** (sequential pre-norm) | `llama`, `phi3`, `qwen2`, `stablelm`, `mistral` | LLaMA-3.2, Qwen-2, Phi-3, Mistral-7B, StableLM-2 |
| **GPT-NeoX-like** (parallel residual) | `gptneox`, `phi2` | Pythia, Phi-2 |
| **Gemma-like** (embedding scaling + post-norm) | `gemma`, `gemma2` | Gemma 2B, Gemma-2 2B/9B |
| **LFM2-like** (hybrid convolution-attention) | `lfm2`, `lfm2_moe` | LFM2 350M–2.6B, LFM2-VL, LFM2-8B-A1B |

Multimodal variants (vision-language via SigLIP2, speech via FastConformer/Mimi) plug in as modality encoders on top of the base LLM backbone — the core runtime is unchanged.

See [doc/model-architectures.md](doc/model-architectures.md) for the full reference.

---

## Architecture at a Glance

```
 ┌────────────────────────────────────────────────────────────────┐
 │  Application  (ASP.NET Core, desktop, CLI, worker service)     │
 └──────────────┬─────────────────────────────────────────────────┘
                │  IChatClient / IAsyncEnumerable<string>
 ┌──────────────▼─────────────────────────────────────────────────┐
 │  Llmdot.Extensions.AI   (Microsoft.Extensions.AI integration)  │
 ├────────────────────────────────────────────────────────────────┤
 │  Llmdot.Core                                                   │
 │  ┌───────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
 │  │ GGUF      │  │ Architecture    │  │ Sampling & Tokenizer │  │
 │  │ Loader    │─▶│ Resolver        │─▶│                      │  │
 │  └───────────┘  └────────┬────────┘  └──────────────────────┘  │
 │                          ▼                                     │
 │                 ┌─────────────────┐                            │
 │                 │ Model Graph     │  4 execution templates     │
 │                 │ + KV / Conv     │  resolved from config      │
 │                 │   State         │                            │
 │                 └────────┬────────┘                            │
 │                          ▼                                     │
 │                 ┌─────────────────┐                            │
 │                 │ Tensor Runtime  │  managed kernels,          │
 │                 │ (IComputeBackend)│  Span<T>, intrinsics      │
 │                 └────────┬────────┘                            │
 └──────────────────────────┼─────────────────────────────────────┘
                            ▼
               ┌────────────┴────────────┐
               │                         │
         ┌─────▼─────┐            ┌──────▼──────┐
         │ CPU       │            │ Optional    │
         │ (default, │            │ GPU: Vulkan │
         │  managed) │            │ / Metal /   │
         │           │            │ CUDA        │
         └───────────┘            └─────────────┘
```

The model graph reads only from a resolved `TransformerConfig` — never from raw GGUF keys directly. This is the central abstraction that eliminates per-architecture code paths. See [doc/architecture.md](doc/architecture.md).

---

## Project Goals

- Load and execute supported GGUF models directly from .NET
- Cover all major 1–8B decoder architectures via the four execution templates
- Support small multimodal models (vision-language, audio) through pluggable modality encoders
- Provide a clean chat and text-generation API with async streaming and cancellation
- Integrate naturally with `Microsoft.Extensions.AI` abstractions
- Deliver strong CPU performance for quantized small-to-mid-sized models
- Offer optional GPU compute backends (Vulkan, Metal) without coupling model support to a vendor format

---

## Non-Goals

- Be the fastest inference engine on every hardware target
- Replace vendor-optimized GPU runtimes for large-scale serving
- Require ONNX conversion or proprietary model packaging
- Target frontier-scale (70B+) models as an early milestone
- Accelerate via NPU — NPUs are graph compilers, not programmable compute. See [architecture.md](doc/architecture.md) for the reasoning.

---

## Packaging

| Package | Purpose | Dependencies |
|---|---|---|
| `Llmdot.Core` | GGUF loader, model graph, CPU backend, sampling, tokenizer | Pure managed .NET |
| `Llmdot.Extensions.AI` | `IChatClient` + `Microsoft.Extensions.AI` integration | `Llmdot.Core` |
| `Llmdot.Backends.Vulkan` *(planned)* | Vulkan compute acceleration | Native Vulkan loader |
| `Llmdot.Backends.Metal` *(planned)* | Metal compute acceleration (Apple Silicon) | Native Metal |
| `Llmdot.Multimodal.Vision` *(planned)* | SigLIP2 vision encoder + connector | `Llmdot.Core` |

The core runtime is the single required dependency. Everything else is additive and opt-in.

---

## Repository Layout

```
llmdot/
├── src/
│   ├── Llmdot.Core/              Core runtime: GGUF loader, graph, CPU backend
│   └── Llmdot.Extensions.AI/     Microsoft.Extensions.AI integration
├── samples/
│   └── Llmdot.Sample/            Minimal end-to-end example
├── tests/
│   └── Llmdot.Core.Tests/        Unit and integration tests
├── benches/
│   └── Llmdot.Benchmarks/        BenchmarkDotNet performance suite
└── doc/                          Vision, architecture, roadmap, platform strategy
```

Target frameworks: **net8.0**, **net9.0**, **net10.0**. `Nullable` enabled, warnings-as-errors, `LangVersion=13.0`.

---

## Status

**Pre-alpha.** The specification, architecture, and execution template design are stable. Implementation is in active development. Do not use in production yet.

Initial release targets:

- [x] Architecture and execution template design
- [ ] GGUF loader (header, metadata, tensors, tokenizer assets)
- [ ] `TransformerConfig` resolver across all four templates
- [ ] CPU reference backend with quantized kernels
- [ ] LLaMA-like template end-to-end
- [ ] Token streaming via `IAsyncEnumerable<T>`
- [ ] `IChatClient` integration
- [ ] Remaining three execution templates
- [ ] Optional GPU backends

Track progress in [doc/roadmap.md](doc/roadmap.md).

---

## Contributing

This is an early-stage project and design feedback is welcome. Please read the [vision](doc/vision.md) and [architecture](doc/architecture.md) documents before opening an issue — most "why not X?" questions have explicit answers there (especially around ONNX, NPU, and native wrapping).

Contribution areas most valuable right now:

- GGUF quantization format coverage
- Managed kernel optimization (intrinsics, vectorization)
- Tokenizer correctness across BPE variants
- Test fixtures for additional model families

---

## Guiding Principle

> `llmdot` should aim to become the easiest way to run community GGUF models from .NET:
>
> - one core package to get started
> - one model format
> - one programming model
>
> Performance still matters, but friction reduction is the primary product advantage.

---

## License

MIT. See [LICENSE](LICENSE) for details.
