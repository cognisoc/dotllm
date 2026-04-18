# Architecture

## Overview

`llmdot` uses a layered architecture that separates model loading, execution orchestration, and hardware acceleration. The core runtime is pure managed code. Backend adapters accelerate a narrow set of expensive primitives when available.

The runtime supports major decoder-only transformer and hybrid architectures in the 1-8B parameter range through **four execution templates** and a unified `TransformerConfig` resolved from GGUF metadata at load time. No per-model code paths are needed — all variation is expressed through configuration.

## High-Level Flow

1. Load a GGUF model and parse metadata, tensor layout, tokenizer assets, and quantization details.
2. Resolve a `TransformerConfig` from GGUF metadata keys and tensor names.
3. Select the appropriate execution template based on config flags.
4. Execute token generation through a managed inference loop.
5. Dispatch expensive tensor operations to the selected compute backend.
6. Stream decoded tokens through idiomatic .NET APIs.

## Supported Architectures

| `general.architecture` | Model Families | Template |
|---|---|---|
| `llama` | LLaMA-1/2/3, Mistral-7B, TinyLlama-1.1B, Mixtral | LLaMA-like |
| `phi3` | Phi-3 Mini/Medium | LLaMA-like |
| `qwen2` | Qwen-2 1.5B/7B | LLaMA-like |
| `phi2` | Phi-2 2.7B | GPT-NeoX-like |
| `gptneox` | Pythia 1.4B/6.9B | GPT-NeoX-like |
| `gemma` | Gemma 2B | Gemma-like |
| `gemma2` | Gemma-2 2B/9B | Gemma-like |
| `stablelm` | StableLM-2 1.6B | LLaMA-like (with flags) |
| `lfm2` | LFM2 350M/700M/1.2B/2.6B, LFM2-VL | LFM2-like |
| `lfm2_moe` | LFM2-8B-A1B, LFM2-24B-A2B | LFM2-like (MoE) |

### LFM2 Architecture

LFM2 (Liquid Foundation Model 2 by Liquid AI) is a **hybrid convolution-attention architecture** purpose-built for on-device inference. It replaces ~62% of attention layers with double-gated LIV (Liquid Input-Varying) causal convolutions, yielding significantly faster CPU inference than pure transformers of comparable size.

**Key properties:**
- Hybrid layer layout: most layers are gated convolutions (kernel=3), only ~38% are GQA attention layers
- GQA: 16/32 Q heads, 8 KV heads (2:1 ratio), RoPE with θ=1,000,000
- SwiGLU FFN with auto-adjusted intermediate dimension
- Vocab: 65,536 (custom BPE with ChatML-style special tokens)
- MoE variants: 32 experts, top-4 routing (8B-A1B, 24B-A2B)
- Context: 32K

**Multimodal variants:**
- **LFM2-VL**: Vision-language (SigLIP2 vision encoder + LFM2 backbone + 2-layer MLP connector)
- **LFM2-Audio**: Speech-to-speech (FastConformer encoder + Mimi codec decoder)

The LFM2 template introduces a new structural primitive — the **gated convolution block** — that does not exist in the other three templates. This is the reason it requires its own execution template rather than being expressed through the LLaMA-like path.

## Four Execution Templates

All supported architectures collapse into four execution templates. Within each template, all variation is expressed through `TransformerConfig` values.

### Template 1: LLaMA-like (sequential pre-norm)

Covers: `llama`, `phi3`, `qwen2`, `stablelm`, `mistral`

```
x → norm → attn → + → norm → ffn → + → output
                  ↑                  ↑
                  x                  x
```

- Pre-norm placement (norm before attention and FFN)
- Separate Q/K/V tensors (or fused, handled by TensorNameResolver)
- SwiGLU or GeGLU FFN (detected by presence of gate tensor)
- No post-norm by default

### Template 2: GPT-NeoX-like (parallel residual)

Covers: `gptneox`, `phi2`

```
x → norm → attn → + → output
                  ↑
       norm → ffn ┘
                  ↑
                  x
```

- Parallel residual: attention and FFN compute independently from the same input, results are summed
- Fused QKV tensor (`attn_qkv` instead of separate Q/K/V)
- Partial rotary embeddings (`rope.dimension_count` < `head_dim`)
- May use LayerNorm or RMSNorm (determined from metadata)

### Template 3: Gemma-like (embedding scaling + post-norm)

Covers: `gemma`, `gemma2`

```
x → scale → norm → attn → + → post_norm → norm → ffn → + → post_norm → output
                          ↑                                 ↑
                          x                                 x
```

- Embedding scaling: `√hidden_size` applied after token embedding lookup
- Post-norm: RMSNorm applied *after* the residual addition (not before)
- Gemma-2 adds logit softcapping and interleaved sliding window attention
- Tied embeddings (`output.weight` absent; `token_embd.weight` used for logits)

### Template 4: LFM2-like (hybrid convolution-attention)

Covers: `lfm2`, `lfm2_moe`

```
Per layer (attention):
  x → norm → attn → + → norm → ffn → + → output
                    ↑                  ↑
                    x                  x

Per layer (convolution):
  x → gate_down → gate_up → conv1d → gate → + → output
                                               ↑
                                               x
```

- Mixed layer types: attention layers and gated convolution layers alternate in a fixed pattern described by metadata
- Convolution blocks: double-gated short-range causal convolutions (kernel=3) with SwiGLU-style gating
- Attention layers use GQA with RoPE
- SwiGLU FFN in attention layers
- Layer type is determined per-layer by the presence of attention tensors vs convolution tensors
- MoE variants route FFN computation through expert selection

**New primitives required by this template:**
- 1D causal convolution (kernel=3) — a simple sliding dot product, not a full conv framework
- Double-gated activation: `gate1 * silu(gate2 * input) * conv1d(...)`
- These are implemented as tensor runtime primitives, not as a separate subsystem

## Core Subsystems

### GGUF Loader

The loader is responsible for:

- reading GGUF headers and tensor metadata
- validating model architecture and quantization support
- mapping tensors into a runtime-friendly structure
- exposing tokenizer and generation metadata to higher layers

Unsupported model features must fail clearly and early.

### Architecture Resolver

The architecture resolver reads `general.architecture` and GGUF metadata keys to produce a fully resolved `TransformerConfig`. This is the central abstraction that eliminates per-architecture code paths.

Key resolving logic:

| Dimension | Resolution strategy |
|---|---|
| **Norm type** | `layer_norm_rms_epsilon` present → RMSNorm; `layer_norm_epsilon` present → LayerNorm |
| **Attention grouping** | `head_count == head_count_kv` → MHA; `head_count_kv == 1` → MQA; else → GQA |
| **FFN type** | `ffn_gate` tensor present → SwiGLU/GeGLU; absent → standard MLP |
| **QKV layout** | `attn_qkv` tensor present → fused; `attn_q`/`attn_k`/`attn_v` → separate |
| **Tied embeddings** | `output.weight` tensor absent + `token_embd.weight` present → tied |
| **Parallel residual** | `use_parallel_residual` key → bool |
| **Post-norm** | `post_attention_norm`/`post_ffw_norm` tensors present → post-norm active |
| **Embedding scaling** | `embedding_scale` key or architecture-specific default (gemma → √hidden_size) |
| **Sliding window** | `attention.sliding_window` > 0 → SWA active |
| **Logit softcapping** | `attn_logit_softcapping`/`final_logit_softcapping` keys present |
| **MoE** | `expert_count` > 0 → MoE active |
| **Hybrid conv-attn** | `lfm2`/`lfm2_moe` architecture → LFM2-like template; per-layer type determined by tensor presence |

For LFM2 models, the per-layer type (convolution vs attention) is resolved by inspecting each layer's tensors: if attention tensors (Q/K/V or QKV) are present, it is an attention layer; otherwise it is a convolution layer.

### TransformerConfig

A fully resolved POCO that captures all hyperparameters from GGUF metadata. The model graph reads only from this config — never from raw GGUF keys directly.

```csharp
public sealed class TransformerConfig
{
    // Architecture identity
    public string Architecture { get; init; }
    public ExecutionTemplate Template { get; init; }

    // Model dimensions
    public int HiddenSize { get; init; }
    public int LayerCount { get; init; }
    public int ContextLength { get; init; }
    public int VocabSize { get; init; }
    public int FfnDim { get; init; }

    // Attention
    public AttentionType AttentionType { get; init; }
    public int HeadCount { get; init; }
    public int HeadCountKv { get; init; }
    public int HeadDim { get; init; }
    public QkvLayout QkvLayout { get; init; }

    // Normalization
    public NormType NormType { get; init; }
    public float NormEpsilon { get; init; }
    public bool HasPostNorm { get; init; }

    // Positional encoding
    public float RopeFreqBase { get; init; }
    public int RopeDimensionCount { get; init; }
    public RopeScalingType RopeScalingType { get; init; }
    public float RopeScalingFactor { get; init; }

    // FFN
    public FfnType FfnType { get; init; }

    // Structural
    public bool ParallelResidual { get; init; }
    public bool TiedEmbeddings { get; init; }
    public float EmbeddingScale { get; init; }

    // Attention features
    public int SlidingWindow { get; init; }
    public float? AttnLogitSoftcap { get; init; }
    public float? FinalLogitSoftcap { get; init; }

    // MoE
    public int ExpertCount { get; init; }
    public int ExpertUsedCount { get; init; }
    public int ExpertFfnDim { get; init; }

    // Hybrid conv-attention (LFM2)
    public bool HasConvLayers { get; init; }
    public int ConvKernelSize { get; init; }
    public LayerType[] LayerTypes { get; init; }
}
```

The `LayerTypes` array stores the resolved type for each layer (Attention or Conv), enabling the LFM2 template to dispatch to the correct execution path per-layer without inspecting tensors at inference time.

### Tensor Name Resolver

Tensor names differ between architectures. Rather than hardcoding per-architecture lookups, a `TensorNameResolver` provides a uniform access layer:

- Checks for fused tensors first (`attn_qkv`), falls back to separate (`attn_q`, `attn_k`, `attn_v`)
- Checks for gated FFN (`ffn_gate` + `ffn_up`) vs standard (`ffn_up` only)
- Handles Gemma post-norm tensors gracefully (absent = no post-norm)
- Handles LFM2 convolution tensors (`conv1d`, `gate_down`, `gate_up`) vs attention tensors per layer
- Returns `null` or a sentinel for absent optional tensors rather than throwing

This makes the runtime tolerant of tensor naming variations without per-architecture switch statements.

### Model Graph

The model graph implements the four execution templates. It defines:

- token embeddings (with optional scaling)
- attention blocks (MHA/GQA/MQA, optional sliding window, optional softcapping)
- convolution blocks (gated causal conv1d, for LFM2)
- feed-forward layers (SwiGLU/GeGLU/standard, optional MoE)
- normalization layers (RMSNorm/LayerNorm, pre-norm/post-norm)
- logits projection (with optional softcapping, tied embeddings)

The execution order is determined by the template plus config flags, not by the architecture string. This layer owns the state transitions for generation.

### Tensor Runtime

The tensor runtime is the numerical execution substrate for the managed core. It should provide:

- memory layout abstractions
- quantized tensor access patterns
- vectorized math primitives
- 1D causal convolution (kernel=3, for LFM2 conv blocks)
- buffer reuse and pooling strategies
- shape and stride utilities for inference operations

This is where low-allocation execution discipline matters most.

### KV Cache and State Management

Efficient local inference depends on a well-defined cache strategy. The runtime should provide:

- reusable key/value cache storage sized per `HeadCountKv` (not `HeadCount`)
- prompt prefill and decode-phase separation
- predictable lifecycle management for sessions
- bounded memory behavior for long-running chats
- sliding window support: cache entries beyond the window can be evicted
- for LFM2: convolution layers maintain a fixed-size recurrence state (kernel=3 sliding window) rather than growing KV cache

### Multimodal Integration

The runtime should support multimodal model variants that embed additional modality inputs into the token stream:

- **Vision-language (VL):** Vision encoder outputs are projected through a connector and inserted as token embeddings at designated positions (e.g., `<image>` sentinel tokens). The LLM backbone processes them identically to text tokens.
- **Audio:** Audio encoder outputs are similarly projected and interleaved.

The architecture supports this by:
1. Loading the base LLM backbone from the GGUF file (same as text-only)
2. Routing special tokens (e.g., `<image>`) to a modality-specific encoder
3. Replacing sentinel token embeddings with projected visual/audio features before the backbone forward pass

Modality encoders are pluggable and loaded separately from the base GGUF model. The core runtime focuses on the LLM backbone; multimodal support is additive.

### Sampling Engine

The sampling layer should support:

- greedy decoding
- top-k and top-p sampling
- temperature scaling
- repetition penalties
- deterministic seeded generation when requested

Sampling should remain independent from the compute backend.

### Tokenizer

GGUF embeds tokenizer data as metadata arrays. The runtime should support:

- BPE tokenizers (llama-style, used by most architectures)
- SentencePiece-based tokenizers (overlap with BPE encoding)
- tiktoken-based tokenizers (used by Qwen-2, LLaMA-3)
- Custom BPE with extended special tokens (used by LFM2)

The `tokenizer.ggml.model` metadata key determines the encoding algorithm. Token strings and merge rules are read from the corresponding metadata arrays.

### API Surface

The public API should be idiomatic to .NET:

- `IAsyncEnumerable<T>` token streaming
- cancellation-aware generation calls
- dependency injection registration helpers
- `Microsoft.Extensions.AI` integration through `IChatClient`

The API should avoid exposing backend-specific assumptions in common application code.

## Compute Backend Abstraction

The compute backend abstraction is the key extensibility point. The model graph should target a small backend contract rather than vendor APIs directly.

### Backend Strategy: GPU Compute, Not NPU

**NPUs are not a viable acceleration target for LLM inference today.** The reasoning:

1. **NPUs are graph execution engines, not programmable compute devices.** You cannot dispatch individual MatMul or RoPE operations; you must compile a subgraph with static shapes. This is fundamentally incompatible with the "thin backend adapter" pattern.

2. **NPU memory = system RAM.** NPUs share memory bandwidth with the CPU. LLM inference at batch=1 is bandwidth-bound, so the NPU provides no throughput advantage over CPU for weight loading. The bottleneck is identical.

3. **NPU requires model format conversion.** All NPU APIs (DirectML, QNN, CoreML) require ONNX or vendor-specific graph IR. This defeats the GGUF-native value proposition.

4. **No major LLM framework targets NPU.** llama.cpp, MLX, and vLLM all target programmable GPU compute (Vulkan, Metal, CUDA). This is a correct technical judgment, not an oversight.

5. **NPU op coverage is insufficient.** No RoPE, no KV cache management, no custom attention patterns, no fused dequant+matmul on any NPU platform.

**The right acceleration target is GPU compute via programmable shaders**, where:
- Custom kernels for fused dequant+matmul are possible
- GGUF weights are used directly (no format conversion)
- Real memory bandwidth advantage exists (GPU VRAM >> system RAM on discrete GPUs; unified memory on Apple Silicon)
- The thin backend adapter pattern maps naturally

### Backend Contract

The backend contract defines a minimal set of tensor primitives that the model graph calls into:

- Matrix multiplication (quantized × quantized, quantized × float)
- RoPE application (in-place on key/value tensors)
- Softmax (with optional softcapping)
- RMSNorm / LayerNorm
- Elementwise operations (SiLU, GeLU, addition, scaling)
- Scaled dot-product attention (with optional causal mask and sliding window)
- 1D causal convolution (for LFM2 conv blocks)
- Argmax / sampling prep

All operations take `ReadOnlySpan<T>` inputs and `Span<T>` outputs. The backend never owns model weights — it borrows them.

### Default Backend

The default backend is CPU-based and entirely managed. It should use:

- `Span<T>` and `Memory<T>` for safe low-level access
- hardware intrinsics where available (AdvSimd on ARM, Vector256 on x64)
- vectorized kernels for quantized inference hot paths

The CPU backend is the reference implementation and must remain fully functional without optional dependencies.

### Optional GPU Backends

Optional backends accelerate expensive primitives via GPU compute:

| Backend | Platform | .NET Access | Format |
|---|---|---|---|
| **Vulkan compute** | Cross-platform (Windows, Linux, Android) | P/Invoke or Silk.NET | GGUF direct |
| **Metal compute** | macOS / iOS (Apple Silicon) | P/Invoke (ObjC runtime) | GGUF direct |
| **CUDA** | NVIDIA (optional, later) | P/Invoke or ManagedCuda | GGUF direct |

These backends implement the same `IComputeBackend` contract. They can offload individual operations (e.g., just MatMul) or entire layer computations. The model graph does not need to know which backend is active.

The accelerator path should focus on a small set of high-value operations:

- matrix multiplication (the dominant cost)
- attention-related projection work
- softmax and related reduction-heavy operations

This keeps hardware offload incremental rather than all-or-nothing.

### NPU: Watch and Revisit

NPU acceleration may become viable in the future if:
- WebNN matures and provides a standard compute API reaching NPU hardware
- Intel/Qualcomm/Apple expose programmable NPU compute (not just graph compilation)
- NPU op coverage expands to include RoPE, dynamic KV cache, and custom attention
- A standard "NPU compute shader" API emerges

None of these are likely in the next 2-3 years. If they materialize, the path would be a separate `Llmdot.Backends.Ort` package using ONNX Runtime with platform-specific execution providers — but this requires GGUF-to-ONNX conversion and is therefore an all-or-nothing acceleration model, not the incremental offload pattern.

## Packaging Strategy

The project should preserve a small, comprehensible package structure. The core runtime package is the single required dependency for getting started; additional packages are optional and additive:

- **core runtime package** for GGUF loading and CPU inference
- **optional integration package** for higher-level chat abstractions and `IChatClient` support
- **optional backend packages** for GPU hardware acceleration (Vulkan, Metal) — these may introduce platform-specific native dependencies
- **optional multimodal packages** for vision-language and audio extensions — these include modality encoders and connectors

This keeps the default install path as a single package while allowing advanced deployments to opt in to more capability.

## Performance Strategy

The performance target is not absolute leadership. The target is practical throughput for local quantized models on common hardware.

Performance work should prioritize:

- quantized CPU kernels (dequantize-through-multiply fusion for the hot path)
- minimizing allocations in the generation loop
- efficient cache reuse (GQA/MQA share KV across Q heads)
- convolution block optimization for LFM2 (small fixed-size state vs growing KV cache)
- short startup and model load times
- acceptable latency for 1B to 8B scale models

## Risks

The main architectural risks are:

- unsupported GGUF variants and quantization schemes
- complexity in maintaining efficient managed kernels across quantization types
- backend abstraction growing too broad too early
- platform-specific GPU accelerator integrations becoming brittle
- partial rotary embeddings requiring careful boundary handling in RoPE kernels
- LFM2 hybrid conv-attention introducing a new execution path with its own optimization surface
- NPU ecosystem fragmentation if hardware vendors diverge on API strategy

These risks are best managed by keeping the first releases CPU-first, supporting the four established execution templates, and being explicit about supported model families and quantization formats.