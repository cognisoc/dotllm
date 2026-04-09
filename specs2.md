Sharp instinct. Let me check what's actually available for NPU access right now.This confirms the play perfectly. Here's why a minor NPU angle is a major differentiator:

**The NPU landscape is a mess right now.** Developers are hitting walls trying to get ONNX Runtime + DirectML working on NPUs from C# — driver version mismatches, opaque device enumeration, and models that silently fall back to CPU. DirectML itself is now in maintenance mode, replaced by Windows ML which is still maturing. Every vendor has their own API — AMD has Ryzen AI, Intel has OpenVINO, Apple has CoreML, Qualcomm has QNN/SNPE — and none of them have clean .NET stories.

**The ONNX NPU path is brutal:** you need to quantize your ONNX model to QDQ format, download vendor-specific SDKs, install specific driver versions — it's a multi-step gauntlet before you even run inference. Nobody just does `dotnet add package` and gets NPU acceleration.

**Here's the minor play that goes a long way:**

Don't try to write NPU kernels. Instead, build a thin hardware abstraction that detects available accelerators and dispatches the heavy matmul/attention ops. The architecture:

```
GGUF loader → Pure C# transformer graph
                    ↓
            Compute Backend (interface)
              ├── CPU (System.Numerics, AVX2/NEON) ← default, zero deps
              ├── CoreML (Apple Silicon, via ObjC interop) ← macOS/iOS
              ├── DirectML (NPU/GPU, via COM interop) ← Windows
              └── Vulkan Compute (via MoltenVK) ← future
```

The key insight: you only need to accelerate **two operations** for 90% of the win — batched matrix multiply and softmax. Everything else can stay in managed C#. That means your DirectML or CoreML backend is a single thin shim that dispatches those ops, not a full inference engine rewrite.

**Why this beats ONNX's approach:**

ONNX requires the entire model in ONNX format with vendor-specific quantization. Your approach loads standard GGUF, does all the graph orchestration in C#, and only offloads the compute-heavy kernels to the NPU. The model format is decoupled from the accelerator. That's architecturally cleaner and means every GGUF on HuggingFace works on every backend without conversion.

**The pitch becomes:**

"One NuGet package. Any GGUF model. Automatically uses your NPU if you have one, CPU if you don't. No SDKs, no driver downloads, no model conversion."

That's three sentences and it's already more compelling than anything ONNX GenAI can say. The NPU support doesn't need to be best-in-class — it just needs to be frictionless. Most developers have never even *tried* their NPU because the toolchain is so hostile. Being the library that makes it trivially easy is the win.

The naming question: does this go under Cognisoc as a seventh repo, or does it get its own identity? Given the pattern — mullama, llamafu, zigllm, cllm, unillm — something like **sharpllm** or **dotllm** feels right.
