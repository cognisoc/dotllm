# Roadmap

## Roadmap Principles

The roadmap should favor a usable vertical slice over broad early ambition. `llmdot` becomes credible when it can load a real GGUF model, run a complete generation loop, and integrate cleanly into ordinary .NET applications.

## Phase 0: Foundation (COMPLETE)

Goals:

- establish repository structure and package boundaries
- define supported runtime targets
- document coding and benchmarking conventions
- agree on the first supported model family and quantization formats

Decisions:

- **Target frameworks:** net8.0, net9.0, net10.0 (net8.0 LTS is the compatibility floor)
- **Supported architectures:** All major 1-8B model families via four execution templates:
  - LLaMA-like: `llama`, `phi3`, `qwen2`, `stablelm` (sequential pre-norm, SwiGLU)
  - GPT-NeoX-like: `gptneox`, `phi2` (parallel residual, fused QKV, partial rotary)
  - Gemma-like: `gemma`, `gemma2` (embedding scaling, post-norm, softcapping)
  - LFM2-like: `lfm2`, `lfm2_moe` (hybrid convolution-attention, on-device optimized)
- **Multimodal scope:** Small multimodal models supported via pluggable modality encoders (vision-language, audio) on top of the base LLM backbone. Sentinel tokens in the tokenizer trigger modality-specific processing.
- **Quantization formats:** All common types — block quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), K-quants (Q2_K through Q6_K), and F16/F32/BF16
- **Key abstraction:** `TransformerConfig` resolved from GGUF metadata at load time eliminates per-architecture code paths. Variation is expressed through config values, not conditional branches on architecture strings.
- **Acceleration strategy:** GPU compute (Vulkan, Metal) is the target for Phase 4. NPU is not viable for LLM inference today — NPUs are graph compilers, not programmable compute; they share system RAM bandwidth with CPU; they require ONNX conversion. See architecture doc for full analysis.
- **Project structure:**
  - `src/Llmdot.Core` — core runtime (GGUF loader, tensors, model graph, sampling, inference)
  - `tests/Llmdot.Core.Tests` — xunit test project
  - `benches/Llmdot.Benchmarks` — BenchmarkDotNet harness
  - `samples/Llmdot.Sample` — console sample app
- **Coding conventions:** C# 13, nullable enabled, warnings as errors, unsafe blocks allowed in core, .editorconfig enforced, `InternalsVisibleTo` for test and extension projects

Exit criteria:

- [x] architecture baseline is documented
- [x] benchmark harness plan exists
- [x] model scope is frozen (all 1-8B architectures via 4 templates, including multimodal variants)

## Phase 1: Managed Core Runtime

Goals:

- implement GGUF parsing for all supported architectures
- implement `TransformerConfig` resolution from GGUF metadata
- implement `TensorNameResolver` for architecture-tolerant tensor access
- build the managed tensor and buffer abstractions
- implement the four execution templates (LLaMA-like, GPT-NeoX-like, Gemma-like, LFM2-like)
- implement token generation on a CPU-only backend
- support basic sampling and streaming APIs
- implement BPE tokenizer decoding from GGUF metadata
- add 1D causal convolution primitive for LFM2 conv blocks

Exit criteria:

- a supported GGUF model from each template can be loaded directly
- the runtime can generate text end-to-end on CPU for at least one model per template
- `TransformerConfig` is correctly resolved from real GGUF files
- token streaming works through an idiomatic public API

### Phase 1 Milestones (ordered)

1. **GGUF metadata → TransformerConfig:** Parse all architecture-specific metadata keys and resolve a complete `TransformerConfig` including norm type, attention type, FFN type, QKV layout, parallel residual flag, embedding scaling, MoE presence, and hybrid conv-attention layer types.
2. **Tensor name resolution:** Implement `TensorNameResolver` that handles fused vs separate QKV, gated vs standard FFN, tied vs separate embeddings, optional post-norm tensors, and LFM2 convolution tensors per layer.
3. **Tensor runtime primitives:** Implement quantized dequantization, matrix multiplication (quantized × float), RoPE, RMSNorm, LayerNorm, SiLU, GeLU, softmax, and 1D causal convolution — all on CPU with `Span<T>`.
4. **LLaMA-like template:** Implement the sequential pre-norm execution path. Validate with LLaMA-3.2-1B or Qwen-2-1.5B.
5. **GPT-NeoX-like template:** Add parallel residual and fused QKV path. Validate with Phi-2 or Pythia-1.4B.
6. **Gemma-like template:** Add embedding scaling, post-norm, and softcapping. Validate with Gemma-2B.
7. **LFM2-like template:** Add hybrid conv-attention with per-layer type dispatch, double-gated conv blocks, and GQA attention. Validate with LFM2-1.2B.
8. **Sampling and streaming:** Integrate the sampling engine and expose `IAsyncEnumerable<T>` token streaming.
9. **Tokenizer decoding:** Implement BPE decode from GGUF metadata arrays (`tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.merges`). Support custom BPE with extended special tokens (LFM2).

## Phase 2: Developer-Facing API (COMPLETE)

Goals:

- stabilize the public API surface
- add dependency injection registration
- add `IChatClient` integration for `Microsoft.Extensions.AI`
- improve error reporting and model capability inspection
- validate all supported architectures against real GGUF files
- add sentinel token routing for multimodal inputs (image, audio)
- add chat template formatting from GGUF metadata

Exit criteria:

- a standard .NET app can consume `llmdot` without custom plumbing
- chat and text generation scenarios are documented
- unsupported models fail with useful diagnostics
- all four templates produce correct output for their reference models
- multimodal models can accept image inputs through a pluggable encoder interface

### Phase 2 Milestones

1. **Microsoft.Extensions.AI integration:** `Llmdot.Extensions.AI` project with `LlmdotChatClient` implementing `IChatClient`, `LlmdotOptions` configuration class, and `AddLlmdot()` DI registration. Streaming support via `GetStreamingResponseAsync`.
2. **Chat template formatting:** `ChatTemplate` class parses `tokenizer.chat_template` from GGUF metadata and formats messages for ChatML, Llama-3, Gemma, Phi-3, and Llama-2 templates. Falls back to a simple generic format for unknown templates.
3. **Public API surface audit:** All implementation-detail types made `internal` with `InternalsVisibleTo` for tests and extensions. Public API limited to: `InferenceEngine`, `LoadedModel`, `ChatSession`, `TransformerConfig`, `GenerationOptions`, `SamplingOptions`, `ModelCapabilities`, `BpeTokenizer`, `ChatTemplate`, `ChatMessageEntry`, and architecture enums.
4. **Model capability inspection:** `ModelCapabilities` class exposes architecture, template, attention type, MoE status, sliding window, softcapping from `TransformerConfig`.
5. **High-level ChatSession:** String-in/string-out generation wrapper with conversation history and BOS token handling.

## Phase 3: Performance and Reliability (IN PROGRESS)

Goals:

- optimize quantized CPU kernels (dequantize-through-multiply fusion)
- reduce allocations and warm-up overhead
- add benchmark suites and regression gates
- validate behavior across Windows, Linux, and macOS
- optimize GQA/MQA KV cache sharing
- optimize LFM2 convolution blocks (fixed-size recurrence state, no growing cache)

### Phase 3 Milestones

1. **SIMD vectorization of hot-path math kernels:** Add `System.Numerics.Vector` acceleration to `RmsNorm`, `LayerNorm`, `Softmax` (partial — max-reduction + normalize), `Add`, `Scale`, `Mul` in `VectorMath`. Silu/Gelu remain scalar due to transcendental functions.
2. **Weight dequantization caching:** `LoadedModel.GetDequantizedWeights()` caches norm and output projection weights at first use, eliminating per-token allocation and dequantization overhead (previously ~64+ allocations per token for a 32-layer model).
3. **TopK/TopP sampling:** `SampleToken` now implements TopK filtering before softmax and TopP (nucleus) filtering after softmax, matching the `SamplingOptions` parameters that were previously defined but unused.
4. **Buffer pooling across Generate calls:** `KvCache` and `InferenceBuffers` are now per-engine fields (not per-call allocations), reset/reused across calls. `KvCache` no longer implements `IDisposable` since it only holds managed arrays.
5. **RoPE frequency precomputation:** `InferenceEngine` precomputes the RoPE frequency table at construction time, eliminating repeated `MathF.Pow` calls per head per layer per token.
6. **HalfHelper bit-manipulation FP16 conversion:** Replace `MathF.Pow`-based half-to-float conversion with direct IEEE 754 bit manipulation, eliminating transcendental function calls in dequantize hot paths.
7. **Sampling allocation elimination:** Replace `logits.ToArray()` with `stackalloc` for typical vocab sizes and in-place `Scale`, avoiding per-token heap allocation.
8. **IComputeBackend completeness:** Extended `IComputeBackend` interface with all tensor operations used by `InferenceEngine` (`Add`, `Scale`, `Mul`, `SiluInPlace`, `Softcap`, `Conv1D`, `DequantizeToFloat`, `GeluScalar`), enabling future GPU backends to implement the full compute contract.
9. **BenchmarkDotNet suite:** Added `VectorMathBenchmarks`, `DequantizeBenchmarks`, and `SamplingBenchmarks` measuring hot-path operations at representative sizes (512, 2048, 4096 elements; 32K and 128K vocab).

### Remaining Phase 3 Work

- quantized MatMul dequantize-through-multiply fusion (batched row dequantization)
- MatMulF32 column-major cache optimization / tiling
- GQA/MQA KV cache sharing (multiple query heads share the same KV slots)
- LFM2 conv block recurrence state optimization
- cross-platform validation (Windows, Linux)

Exit criteria:

- CPU throughput is competitive for the intended local-use model range
- benchmark baselines are repeatable
- core runtime is stable across supported operating systems
- LFM2 models show expected CPU speedup over pure-attention architectures

## Phase 4.5: Real Model Readiness

Goals:

- support the most common GGUF quantization formats (K-quants, BF16)
- fix BPE tokenizer to perform actual merge operations
- support GGUF v2 files
- implement RoPE scaling (Linear) and sliding window attention
- create a working sample app for end-to-end inference

### Completed

1. **K-quant dequantization:** Full support for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K super-block formats. All 6 formats implemented with correct scale/bit extraction per the GGUF spec.
2. **BF16 dequantization:** `DequantizeBF16` converts bfloat16 to float32 via direct bit manipulation (shift byte pair to upper 16 bits of IEEE 754 float).
3. **BPE merge operations:** `BpeTokenizer.Encode()` now performs actual BPE merging after initial tokenization — repeatedly merges the highest-priority adjacent pair until no more merges apply.
4. **GGUF v2 support:** Minimum version requirement lowered from 3 to 2, accepting both v2 and v3 files.
5. **RoPE scaling (Linear):** `PrecomputeRopeFrequencies` divides all frequencies by `RopeScalingFactor` when `RopeScalingType.Linear` is set, enabling context-extended models.
6. **Sliding window attention:** `AttentionForward` now applies `SlidingWindow` constraint — only attends to tokens within the sliding window range.
7. **Sample app:** `Llmdot.Sample` now loads a real GGUF model, prints model capabilities, and runs interactive or single-prompt inference through `ChatSession`.

## Phase 4: Optional GPU Compute Backends (IN PROGRESS)

Goals:

- define a stable compute backend contract (`IComputeBackend`)
- implement one experimental GPU compute backend (Vulkan or Metal)
- keep CPU as the default fallback path
- measure real-world benefit against backend complexity
- ensure backend dispatch works for all four templates including LFM2 conv blocks

### Phase 4 Milestones

1. **Compute backend abstraction:** All tensor operations in `InferenceEngine` now route through `IComputeBackend` instead of calling `VectorMath`/`TensorOps` directly. The interface (`IComputeBackend : IDisposable`) covers: MatMul, MatMulF32, RmsNorm, LayerNorm, ApplyRoPE (both overloads), Softmax, Silu, SiluInPlace, Gelu, GeluScalar, Add, Scale, Mul, Softcap, Conv1D, DequantizeToFloat, ArgMax.
2. **Metal compute backend (GPU dispatch):** `Llmdot.Metal` project with `MetalBackend` implementing `IComputeBackend`. Uses Objective-C runtime P/Invoke (`objc_msgSend`) to call Metal APIs directly from C#. Runtime shader compilation from Metal Shading Language source. GPU dispatch implemented for: RmsNorm, LayerNorm, Softmax, Add (in-place and out-of-place), Scale (in-place and out-of-place), Mul, Silu, SiluInPlace, Gelu, MatMulF32. Operations below a size threshold (256 elements) fall back to CPU. MatMul quantized, ApplyRoPE, DequantizeToFloat, Conv1D, ArgMax remain on CPU. `MetalRuntime.IsAvailable` checks for Metal framework via `NativeLibrary.TryLoad`. `MetalInterop` class manages MTLDevice, MTLCommandQueue, MTLComputePipelineState lifecycle through Objective-C runtime message sends. Persistent and scratch GPU buffer management for weight caching across tokens.
3. **Vulkan compute backend (structurally complete):** `Llmdot.Vulkan` project with `VulkanBackend` implementing `IComputeBackend`. Includes `VulkanInterop` class with full Vulkan P/Invoke bindings (instance, device, command pool, descriptor pool, pipeline layout, shader module, buffer/memory management, command buffer lifecycle). Includes SPIR-V compute shader source for RmsNorm, Softmax, and element-wise ops. Currently delegates all operations to CPU (`VectorMath`/`TensorOps`) — GPU dispatch requires SPIR-V binary compilation and testing on a Vulkan-capable device. `VulkanRuntime.IsAvailable` checks for `vulkan-1.dll` / `libvulkan.so`.
4. **Backend selection:** `BackendFactory.CreateBestAvailable()` tries Metal on macOS, Vulkan on Linux/Windows, falls back to CPU. `InferenceEngine` accepts `IComputeBackend` in its internal constructor.

### Remaining Phase 4 Work

- Implement Metal MatMul quantized compute shader (Q4_0, Q8_0, K-quants)
- Implement Vulkan GPU dispatch (requires SPIR-V binary compilation and Vulkan-capable GPU for testing)
- Persistent weight upload to GPU (upload model weights once, reuse across tokens)
- Pipeline GPU dispatch for dequantize-then-matmul fusion
- Benchmark and measure real acceleration vs CPU baseline

### Why GPU Compute, Not NPU

NPUs are graph execution engines, not programmable compute devices. They cannot dispatch individual MatMul or RoPE operations — they require compiled subgraphs with static shapes, which is fundamentally incompatible with LLM inference (dynamic KV cache, custom attention patterns) and with llmdot's thin backend adapter design. NPUs also share system RAM bandwidth with the CPU, providing no throughput advantage for memory-bound LLM decode. All major LLM frameworks (llama.cpp, MLX) target GPU compute, not NPU. See architecture doc for full analysis.

## Phase 5: Ecosystem and Distribution

Goals:

- publish NuGet packages
- provide end-to-end examples (text generation, vision-language chat)
- add model discovery and validation tooling
- clarify support policy for model families and backends

Exit criteria:

- packages are publishable and documented
- example applications cover common integration patterns including multimodal
- contributor guidance is stable enough for external participation

## Deferred Work

These areas should remain explicitly out of the first implementation wave:

- large-model distributed inference
- full GPU-first serving infrastructure
- wide multi-architecture support beyond the 1-8B target range
- broad accelerator matrix support before the CPU backend is mature
- MoE expert routing (only metadata parsing in Phase 1; execution deferred)
- encoder-decoder and encoder-only architectures
- NPU acceleration (revisit if/when programmable NPU compute APIs emerge)
- on-device ONNX conversion pipeline for NPU dispatch