# Model Architecture Reference

This document catalogs the architectural properties of major open-weight LLM families in the 1-8B parameter range commonly distributed in GGUF format. It is intended to inform the dotllm runtime's model graph abstraction and GGUF metadata parsing.

## GGUF Architecture Key

Every GGUF file identifies its architecture via `general.architecture`. The known values for target model families:

| Model Family | `general.architecture` | llama.cpp enum |
|---|---|---|
| LLaMA / LLaMA-2 / LLaMA-3 | `llama` | `LLM_ARCH_LLAMA` |
| TinyLlama (1.1B) | `llama` | `LLM_ARCH_LLAMA` |
| Mistral 7B / Mixtral | `llama` | `LLM_ARCH_LLAMA` |
| Phi-2 | `phi2` | `LLM_ARCH_PHI2` |
| Phi-3 (mini) | `phi3` | `LLM_ARCH_PHI3` |
| Gemma 2B | `gemma` | `LLM_ARCH_GEMMA` |
| Gemma-2 9B | `gemma2` | `LLM_ARCH_GEMMA2` |
| Qwen-1.5/1.8B | `qwen2` | `LLM_ARCH_QWEN2` |
| Qwen-2 (1.5B, 7B) | `qwen2` | `LLM_ARCH_QWEN2` |
| Pythia (1.4B, 6.9B) | `gptneox` | `LLM_ARCH_GPTNEOX` |
| StableLM-2 | `stablelm` | `LLM_ARCH_STABLELM` |

> **Key insight:** TinyLlama and Mistral 7B share `general.architecture = "llama"` with LLaMA. The architecture string does not uniquely identify a model variant; hyperparameters and tensor shapes differentiate them at runtime.

## Per-Architecture Details

### 1. LLaMA / LLaMA-2 / LLaMA-3 (`llama`)

| Property | LLaMA-1 7B | LLaMA-2 7B | LLaMA-3 8B | LLaMA-3.2 1B | LLaMA-3.2 3B |
|---|---|---|---|---|---|
| Hidden size | 4096 | 4096 | 4096 | 2048 | 3072 |
| Layers | 32 | 32 | 32 | 16 | 28 |
| Norm type | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm |
| Attention | MHA 32h/32kv | GQA 32h/32kv | GQA 32h/8kv | GQA 32h/8kv | GQA 24h/8kv |
| Head dim | 128 | 128 | 128 | 64 | 128 |
| Positional encoding | RoPE 10000 | RoPE 10000 | RoPE 500000 | RoPE 500000 | RoPE 500000 |
| FFN type | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| FFN dim | 11008 | 11008 | 14336 | 8192 | 8192 |
| Tied embeddings | No | No | No | No | No |
| Tokenizer | BPE/SentencePiece | BPE/SentencePiece | tiktoken | tiktoken | tiktoken |
| Vocab size | 32000 | 32000 | 128256 | 128256 | 128256 |
| Special features | — | — | — | — | — |

**GGUF metadata keys (prefixed `{arch}.`):**
- `attention.layer_norm_rms_epsilon` — RMSNorm epsilon (typically 1e-5 or 1e-6)
- `attention.head_count` / `attention.head_count_kv`
- `rope.freq_base` — RoPE base frequency
- `rope.dimension_count` — rotary dimensions (equals head_dim for llama)
- `feed_forward_length`
- `embedding_length` / `block_count`

### 2. Mistral 7B / Mixtral (`llama`)

| Property | Mistral-7B-v0.1 | Mistral-7B-v0.2 | Mixtral-8x7B |
|---|---|---|---|
| Hidden size | 4096 | 4096 | 4096 |
| Layers | 32 | 32 | 32 |
| Norm type | RMSNorm pre-norm | RMSNorm pre-norm | RMSNorm pre-norm |
| Attention | GQA 32h/8kv | GQA 32h/8kv | GQA 32h/8kv |
| Head dim | 128 | 128 | 128 |
| Positional encoding | RoPE 10000 | RoPE 100000 | RoPE 100000 |
| FFN type | SwiGLU | SwiGLU | SwiGLU (per-expert) |
| FFN dim | 14336 | 14336 | 14336 |
| Tied embeddings | No | No | No |
| Tokenizer | BPE/SentencePiece | BPE/SentencePiece | BPE/SentencePiece |
| Vocab size | 32000 | 32000 | 32000 |
| Special features | Sliding window 4096 | Sliding window 32K (effectively full) | Sliding window + MoE (8 experts, top-2) |

**Additional GGUF keys for Mixtral:**
- `expert_count` → 8
- `expert_used_count` → 2
- `expert_feed_forward_length`

**Sliding window attention** is indicated by:
- `{arch}.attention.sliding_window` — window size (0 or absent = full attention)
- Sliding window is always "interleaved" — some layers use SWA, others full attention, controlled by the pattern.

### 3. TinyLlama 1.1B (`llama`)

| Property | Value |
|---|---|
| Hidden size | 2048 |
| Layers | 22 |
| Norm type | RMSNorm pre-norm |
| Attention | GQA 32h/4kv |
| Head dim | 64 |
| Positional encoding | RoPE 10000 |
| FFN type | SwiGLU |
| FFN dim | 5632 |
| Tied embeddings | Yes |
| Tokenizer | BPE/SentencePiece |
| Vocab size | 32000 |
| Special features | — |

### 4. Phi-2 (`phi2`)

| Property | Value |
|---|---|
| Hidden size | 2560 |
| Layers | 32 |
| Norm type | LayerNorm pre-norm (post-norm in original; GGUF treats as pre-norm with parallel residual) |
| Attention | MHA 32h/32kv |
| Head dim | 80 |
| Positional encoding | RoPE 10000 (partial rotary: dim=80, not full head_dim) |
| FFN type | Standard MLP with GeLU |
| FFN dim | 10240 |
| Tied embeddings | No |
| Tokenizer | BPE/SentencePiece |
| Vocab size | 51200 |
| Special features | Parallel residual connections (QKV fused as single tensor `attn_qkv`); partial rotary position embedding |

**GGUF notes:** Phi-2 uses `use_parallel_residual = true` — the FFN and attention paths are computed in parallel from the same input, then summed. The `%s.attention.key_length` is 80 (partial rotary, not full 80). The QKV is fused into a single tensor `blk.N.attn_qkv`.

### 5. Phi-3 Mini 3.8B / 7B (`phi3`)

| Property | Phi-3 Mini 3.8B | Phi-3 Medium 7B (approx) |
|---|---|---|
| Hidden size | 3072 | 4096 |
| Layers | 32 | 32 |
| Norm type | RMSNorm pre-norm | RMSNorm pre-norm |
| Attention | GQA 32h/8kv | GQA 32h/8kv (varies) |
| Head dim | 96 | 128 |
| Positional encoding | RoPE 10000 | RoPE 10000 |
| FFN type | SwiGLU | SwiGLU |
| FFN dim | 8192 | ~14336 |
| Tied embeddings | No | No |
| Tokenizer | BPE/tiktoken | BPE/tiktoken |
| Vocab size | 32064 | 32064 |
| Special features | Original RoPE scaling for long context | — |

**GGUF notes:** Phi-3 uses the same tensor naming as LLaMA-style (separate Q/K/V tensors, not fused QKV). This is architecturally identical to LLaMA but with its own `general.architecture = "phi3"`. It also supports `rope.scaling.original_context_length` and `rope.scaling.type` for long-context scaling.

### 6. Gemma 2B (`gemma`)

| Property | Value |
|---|---|
| Hidden size | 2048 |
| Layers | 18 |
| Norm type | RMSNorm pre-norm |
| Attention | MHA 8h/1kv (MQA) — 8 Q heads, 1 KV head with head_dim=256 |
| Head dim | 256 |
| Positional encoding | RoPE 10000 |
| FFN type | GeGLU (GeLU-gated) |
| FFN dim | 16384 |
| Tied embeddings | Yes (shared input/output with scaling) |
| Tokenizer | BPE/SentencePiece |
| Vocab size | 256000 |
| Special features | Embedding scale factor (~√hidden_size); one KV head (MQA) |

**GGUF notes:** Gemma applies an `embedding_scale` of `sqrt(embedding_length)` to the token embedding output. This is read from `{arch}.embedding_scale` or computed. The shared embedding tensor means `output.weight` is not present; `token_embd.weight` is used for both.

### 7. Gemma-2 2B / 9B (`gemma2`)

| Property | Gemma-2 2B | Gemma-2 9B |
|---|---|---|
| Hidden size | 2304 | 3584 |
| Layers | 26 | 42 |
| Norm type | RMSNorm pre-norm + post-norm | RMSNorm pre-norm + post-norm |
| Attention | GQA 8h/2kv | GQA 16h/4kv (varies) |
| Head dim | 256 | 256 |
| Positional encoding | RoPE 10000 | RoPE 10000 |
| FFN type | GeGLU | GeGLU |
| FFN dim | 9216 | 14336 |
| Tied embeddings | Yes (with scaling) | No |
| Tokenizer | BPE/SentencePiece | BPE/SentencePiece |
| Vocab size | 256000 | 256000 |
| Special features | Post-attention + post-FFN RMSNorm; sliding window interleaved; logit softcapping | Post-attention + post-FFN RMSNorm; sliding window interleaved; logit softcapping |

**Gemma-2 special features:**
- **Post-norm layers:** `post_attention_norm` and `post_ffw_norm` applied *after* the residual add, not before. This is the "post-norm" variant.
- **Logit softcapping:** `final_logit_softcapping` and `attn_logit_softcapping` — clamps logits via a tanh-based soft cap before softmax or output.
  - `{arch}.final_logit_softcapping`
  - `{arch}.attn_logit_softcapping`
- **Sliding window:** Interleaved local/global attention (every other layer uses SWA).
  - `{arch}.attention.sliding_window` — window size (4096 typical)
- **Sliding window pattern:** Some layers use SWA and others use full attention.

### 8. Qwen-2 1.5B / 7B (`qwen2`)

| Property | Qwen-2 1.5B | Qwen-2 7B |
|---|---|---|
| Hidden size | 1536 | 4096 |
| Layers | 28 | 28 |
| Norm type | RMSNorm pre-norm | RMSNorm pre-norm |
| Attention | GQA 12h/2kv | GQA 28h/4kv |
| Head dim | 128 | 128 |
| Positional encoding | RoPE 1000000 | RoPE 1000000 |
| FFN type | SwiGLU | SwiGLU |
| FFN dim | 8960 | 18944 |
| Tied embeddings | No | No |
| Tokenizer | BPE/tiktoken | BPE/tiktoken |
| Vocab size | 151936 | 152064 |
| Special features | Tie embedding is false | — |

**GGUF notes:** Qwen-2 is architecturally identical to LLaMA. The `general.architecture = "qwen2"` distinguishes it primarily for tensor naming and tokenizer handling. The extremely high RoPE freq_base (1M) enables long context out of the box.

### 9. Pythia 1.4B / 6.9B (`gptneox`)

| Property | Pythia 1.4B | Pythia 6.9B |
|---|---|---|
| Hidden size | 2048 | 4096 |
| Layers | 24 | 32 |
| Norm type | LayerNorm pre-norm | LayerNorm pre-norm |
| Attention | MHA 16h/16kv | MHA 32h/32kv |
| Head dim | 128 | 128 |
| Positional encoding | RoPE 10000 (partial rotary) | RoPE 10000 (partial rotary) |
| FFN type | Standard MLP with GeLU | Standard MLP with GeLU |
| FFN dim | 8192 | 16384 |
| Tied embeddings | No | No |
| Tokenizer | BPE/SentencePiece | BPE/SentencePiece |
| Vocab size | 50304 | 50304 |
| Special features | Parallel residual; partial rotary (rotary_pct); QKV fused | Parallel residual; partial rotary; QKV fused |

**GGUF notes:** Pythia uses `gptneox` architecture. Key differences from LLaMA:
- LayerNorm (not RMSNorm) — read from `%s.attention.layer_norm_epsilon`
- Parallel residual — `%s.use_parallel_residual = true`
- Fused QKV tensor (`blk.N.attn_qkv`)
- Partial rotary — `%s.rope.dimension_count` < head_dim (controlled by `rotary_pct`, typically 0.25)
- Standard GeLU MLP (no gating), not SwiGLU

### 10. StableLM-2 1.6B / 12B (`stablelm`)

| Property | StableLM-2 1.6B |
|---|---|
| Hidden size | 2048 |
| Layers | 24 |
| Norm type | RMSNorm pre-norm |
| Attention | GQA 32h/32kv (varies) |
| Head dim | 64 |
| Positional encoding | RoPE 10000 |
| FFN type | SwiGLU |
| FFN dim | 5632 |
| Tied embeddings | No |
| Tokenizer | BPE/SentencePiece |
| Vocab size | 48000 (varies) |
| Special features | QKV fused; partial rotary; parallel residual |

**GGUF notes:** StableLM-2 reads `use_parallel_residual`, has a `rope.dimension_count` that may be less than head_dim, and uses fused QKV like Phi-2 and GPT-NeoX.

## Common GGUF Metadata Keys (per-architecture)

All architecture-specific keys use the `general.architecture` value as prefix. For example, if `general.architecture = "llama"`, then:

| Canonical key | Format | Description |
|---|---|---|
| `{arch}.context_length` | uint32 | Maximum training context length |
| `{arch}.embedding_length` | uint32 | Hidden size (d_model) |
| `{arch}.block_count` | uint32 | Number of transformer layers |
| `{arch}.feed_forward_length` | uint32 | FFN intermediate size |
| `{arch}.attention.head_count` | uint32 | Number of Q heads |
| `{arch}.attention.head_count_kv` | uint32 | Number of KV heads |
| `{arch}.attention.key_length` | uint32 | Head dimension for K (if differs from d_model/n_head) |
| `{arch}.attention.value_length` | uint32 | Head dimension for V |
| `{arch}.attention.layer_norm_rms_epsilon` | float32 | RMSNorm epsilon |
| `{arch}.attention.layer_norm_epsilon` | float32 | LayerNorm epsilon |
| `{arch}.attention.sliding_window` | uint32 | SWA window size (0 = full attention) |
| `{arch}.rope.freq_base` | float32 | RoPE base frequency |
| `{arch}.rope.dimension_count` | uint32 | Number of rotary dimensions |
| `{arch}.rope.scaling.type` | string | "none", "linear", "yarn", "longrope" |
| `{arch}.rope.scaling.factor` | float32 | RoPE scale factor |
| `{arch}.expert_count` | uint32 | Number of MoE experts |
| `{arch}.expert_used_count` | uint32 | Active experts per token |
| `{arch}.expert_feed_forward_length` | uint32 | Per-expert FFN size |
| `{arch}.use_parallel_residual` | bool | Parallel attention + FFN |
| `{arch}.attn_logit_softcapping` | float32 | Attention logit soft cap |
| `{arch}.final_logit_softcapping` | float32 | Final logit soft cap |
| `{arch}.embedding_scale` | float32 | Embedding scaling factor |
| `tokenizer.ggml.model` | string | Tokenizer type: "llama", "gpt2", "gptneo", etc. |
| `tokenizer.ggml.tokens` | array[string] | Token strings |
| `tokenizer.ggml.scores` | array[float] | Token scores |
| `tokenizer.ggml.merges` | array[string] | BPE merges |
| `tokenizer.ggml.bos_token_id` | uint32 | BOS token ID |
| `tokenizer.ggml.eos_token_id` | uint32 | EOS token ID |

## Analysis: Common vs Varying Dimensions

### Universal constants (same across ALL architectures)

| Dimension | Value |
|---|---|
| Pre-norm placement | All architectures use pre-norm (norm before attention/FFN) |
| Decoder-only | All listed models are autoregressive decoder-only |
| RoPE positional encoding | All use RoPE (no ALiBi in this set) |
| Causal masking | All use causal attention masks |

### Varying dimensions requiring abstraction

| Dimension | Variations | GGUF discriminator |
|---|---|---|
| **Norm type** | RMSNorm vs LayerNorm | `attention.layer_norm_rms_epsilon` present → RMSNorm; `attention.layer_norm_epsilon` present → LayerNorm |
| **Attention grouping** | MHA vs GQA vs MQA | `attention.head_count` vs `attention.head_count_kv`; if equal → MHA; if kv=1 → MQA; else → GQA |
| **FFN type** | SwiGLU (gated) vs GeGLU (gated) vs Standard MLP | Fused gate+up tensor (`ffn_gate` + `ffn_up`) → SwiGLU/GeGLU; single `ffn_up` → standard. Activation: SwiGLU vs GeGLU determined by architecture |
| **Parallel residual** | Additive vs Parallel | `use_parallel_residual` key. `true` = compute attn+FFN in parallel from same input, then add both. `false` (default) = sequential |
| **QKV layout** | Fused vs Separate | Tensor names: `blk.N.attn_qkv` (fused) vs `blk.N.attn_q` / `blk.N.attn_k` / `blk.N.attn_v` (separate) |
| **Tied embeddings** | Shared vs separate | If `output.weight` tensor absent and `token_embd.weight` exists, embeddings are tied |
| **RoPE partial** | Full head_dim vs partial | `rope.dimension_count` < `attention.key_length` → partial rotary |
| **Sliding window** | None vs fixed vs interleaved | `attention.sliding_window` > 0 → SWA active |
| **Post-norm** | None vs post-attention/post-FFN | Presence of `blk.N.post_attention_norm` / `blk.N.post_ffw_norm` tensors |
| **Embedding scaling** | None vs scale factor | `embedding_scale` key or hardcoded per architecture |
| **Logit softcapping** | None vs capped | `final_logit_softcapping` and `attn_logit_softcapping` keys |
| **MoE** | Dense vs MoE | `expert_count` > 0 |
| **Tokenizer** | BPE/SentencePiece vs tiktoken | `tokenizer.ggml.model` string + presence of merges vs precompiled charsmap |

## Recommended .NET Abstraction

### 1. `TransformerConfig` — unified hyperparameters from GGUF

```csharp
public sealed class TransformerConfig
{
    // Architecture identity
    public string Architecture { get; init; }       // general.architecture value

    // Model dimensions
    public int HiddenSize { get; init; }             // embedding_length
    public int LayerCount { get; init; }             // block_count
    public int ContextLength { get; init; }          // context_length
    public int VocabSize { get; init; }              // vocab_size or tokenizer list length
    public int FfnDim { get; init; }                 // feed_forward_length

    // Attention
    public int HeadCount { get; init; }              // attention.head_count
    public int HeadCountKv { get; init; }            // attention.head_count_kv
    public int HeadDim { get; init; }                // key_length or computed hiddenSize/headCount

    // Normalization
    public NormType NormType { get; init; }          // RMSNorm or LayerNorm
    public float NormEpsilon { get; init; }          // from rms_eps or eps key

    // Positional encoding
    public float RopeFreqBase { get; init; }         // rope.freq_base
    public int RopeDimensionCount { get; init; }     // rope.dimension_count
    public RopeScalingType RopeScalingType { get; init; }
    public float RopeScalingFactor { get; init; }

    // FFN
    public FfnType FfnType { get; init; }            // SwiGLU, GeGLU, Standard
    public bool QkvFused { get; init; }              // whether attn_qkv is a single tensor

    // Structural
    public bool ParallelResidual { get; init; }      // use_parallel_residual
    public bool TiedEmbeddings { get; init; }        // no output.weight tensor
    public float EmbeddingScale { get; init; }       // embedding_scale (√d for gemma)
    public bool HasPostNorm { get; init; }           // post_attention_norm + post_ffw_norm

    // Attention features
    public int SlidingWindow { get; init; }          // attention.sliding_window (0 = full)
    public float? AttnLogitSoftcap { get; init; }    // attn_logit_softcapping
    public float? FinalLogitSoftcap { get; init; }   // final_logit_softcapping

    // MoE
    public int ExpertCount { get; init; }            // 0 = dense
    public int ExpertUsedCount { get; init; }        // top-k
    public int ExpertFfnDim { get; init; }            // expert_feed_forward_length
}
```

### 2. Enum abstractions

```csharp
public enum NormType { RmsNorm, LayerNorm }
public enum FfnType { SwiGLU, GeGLU, Standard }
public enum RopeScalingType { None, Linear, YaRN, LongRoPE }
public enum AttentionType { MHA, GQA, MQA }
public enum QkvLayout { Fused, Separate }
```

### 3. Architecture-specific factory

A single `ArchitectureResolver` reads `general.architecture` and `TransformerConfig` to select the correct computation graph. Most architectures collapse into the same graph with different config values. **Only the Phi-2 / GPT-NeoX parallel-residual path and Gemma-2 post-norm path require distinct execution paths.**

### Convergence: 3 execution templates cover all listed architectures

| Template | Architectures | Distinguishing properties |
|---|---|---|
| **LLaMA-like** | `llama`, `phi3`, `qwen2`, `mistral` (same as llama), `stablelm` (with flags) | Sequential pre-norm, separate Q/K/V (or fused), SwiGLU, no post-norm |
| **GPT-NeoX / Phi-2** | `gptneox`, `phi2` | Parallel residual, fused QKV, partial rotary, LayerNorm (gptneox) or RMSNorm (phi2), standard MLP (gptneox) or special (phi2) |
| **Gemma-like** | `gemma`, `gemma2` | Embedding scaling, MQA (gemma) / post-norm + SWA + softcapping (gemma2) |

> The LLaMA-like template covers the majority of popular models. The GPT-NeoX template handles legacy architectures. The Gemma template handles the unique embedding scaling and post-norm patterns. Within each template, all variation is expressed through `TransformerConfig` values — no per-model code paths are needed.

### 4. Key abstraction insight: tensor name mapping

Tensor names differ between architectures (e.g., `attn_qkv` vs separate `attn_q`/`attn_k`/`attn_v`). Rather than hardcoding per-architecture tensor lookups, define a `TensorNameResolver` that:

1. Checks for fused tensors first (`attn_qkv`), falls back to separate tensors (`attn_q`, `attn_k`, `attn_v`)
2. Checks for gated FFN (`ffn_gate` + `ffn_up`) vs standard (`ffn_up` only)
3. Handles Gemma post-norm tensors (`post_attention_norm`, `post_ffw_norm`) gracefully (absent = no post-norm)

This makes the runtime tolerant of tensor naming variations without per-architecture switch statements.

### 5. Attention type determined at load time

```csharp
public static AttentionType DetermineAttentionType(int headCount, int headCountKv) =>
    (headCount, headCountKv) switch
    {
        var (h, kv) when h == kv => AttentionType.MHA,
        var (_, 1) => AttentionType.MQA,
        _ => AttentionType.GQA,
    };
```

### 6. Norm type determined by metadata key presence

```csharp
public static (NormType type, float epsilon) DetermineNormType(GgufMetadata meta, string arch)
{
    var rmsKey = $"{arch}.attention.layer_norm_rms_epsilon";
    var lnKey = $"{arch}.attention.layer_norm_epsilon";
    
    if (meta.TryGetValue(rmsKey, out var rmsVal))
        return (NormType.RmsNorm, (float)rmsVal.Value);
    if (meta.TryGetValue(lnKey, out var lnVal))
        return (NormType.LayerNorm, (float)lnVal.Value);
    
    throw new InvalidDataException($"No norm epsilon found for architecture '{arch}'");
}
```