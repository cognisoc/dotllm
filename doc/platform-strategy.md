# Platform Strategy

## Core Position

`dotllm` should be a CPU-first runtime with optional GPU compute acceleration paths. This is a strategic choice, not a temporary limitation. A managed CPU implementation provides the broadest portability, the smallest deployment surface, and the clearest baseline for correctness.

## Why CPU First

CPU-first execution aligns with the primary product promise:

- no mandatory native runtime packaging
- predictable behavior across ordinary .NET environments
- compatibility with local desktop and edge scenarios
- easier debugging and benchmarking

For many 1B to 8B quantized models, CPU inference is already sufficient for prototyping, offline workflows, and embedded assistant features. Hybrid architectures like LFM2, which replace most attention layers with convolutions, are particularly efficient on CPU.

## The Hardware Fragmentation Problem

Local acceleration is fragmented:

- Apple Silicon has GPU compute via Metal and Neural Engine via CoreML
- Windows has GPU via DirectX, NPU via DirectML, and Intel/AMD/Qualcomm vendor SDKs
- Cross-platform GPU portability remains difficult
- NPU APIs vary by vendor and are graph-compilation-oriented, not compute-oriented

Most available runtimes solve this by making the hardware stack the foundation of the product. `dotllm` should do the opposite: keep the inference engine stable and treat acceleration as an optimization layer.

## Why GPU Compute, Not NPU

NPUs are not a viable acceleration target for LLM inference today:

1. **NPUs are graph execution engines.** They accept compiled compute graphs with static shapes, not individual MatMul or RoPE dispatches. This is fundamentally incompatible with the thin backend adapter pattern.

2. **NPU memory = system RAM.** NPUs share memory bandwidth with the CPU. LLM inference at batch=1 is bandwidth-bound (weight loading dominates), so the NPU provides no throughput advantage.

3. **NPU requires format conversion.** All NPU APIs (DirectML, QNN, CoreML, OpenVINO) require ONNX or vendor IR. This contradicts the GGUF-native strategy.

4. **No major LLM framework uses NPU.** llama.cpp, MLX, and vLLM all target programmable GPU compute (Vulkan, Metal, CUDA). This is correct technical judgment, not oversight.

5. **Insufficient op coverage.** No RoPE, no dynamic KV cache, no custom attention, no fused dequant+matmul on any NPU platform.

NPU may become viable if programmable compute APIs emerge (3-5 year horizon). If that happens, the integration path would be a separate backend package using ONNX Runtime — but this is an all-or-nothing model, not incremental offload.

## Backend Strategy

The backend contract should isolate a small set of expensive primitives. This enables hardware acceleration without coupling the project to a full accelerator-native graph runtime. Optional accelerator backends may introduce platform-specific native dependencies; this is acceptable because the core CPU path remains fully managed.

A practical backend progression is:

1. Managed CPU backend as the baseline
2. Vulkan compute backend for cross-platform GPU acceleration
3. Metal compute backend for Apple Silicon
4. Broader backend support only after measurable value is proven

All GPU backends use GGUF weights directly (no format conversion) and implement the same `IComputeBackend` contract with custom compute shaders for fused operations like dequant+matmul.

## Model Format Strategy

Model compatibility should remain centered on GGUF regardless of backend choice.

This means:

- no required ONNX conversion path
- no vendor-specific packaging as the primary experience
- one model acquisition story across all supported platforms

The model format story should stay constant even when compute execution changes.

## Multimodal Strategy

The base runtime focuses on the LLM backbone. Multimodal support (vision-language, audio) is additive:

- Modality encoders (SigLIP2 for vision, FastConformer for audio) are pluggable and loaded separately
- Sentinel tokens in the tokenizer (e.g., `<image>`) trigger modality-specific processing
- Projected encoder outputs replace sentinel embeddings before the backbone forward pass
- No changes to the core inference loop or execution templates

This keeps the runtime simple while enabling models like LFM2-VL and future multimodal GGUF models.

## Product Differentiation

The platform strategy supports a simple product message:

- supported GGUF models (expanding over time)
- one .NET-first programming model
- CPU by default
- optional GPU acceleration when available
- multimodal as an additive capability

The differentiation is ease of use and consistency, not just hardware throughput.