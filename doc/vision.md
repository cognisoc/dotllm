# Product Vision

## Overview

`llmdot` is a .NET-native local inference runtime for GGUF language models. Its core promise is simple: developers should be able to download a community model, reference a single core package, and run inference in a standard .NET application without managing native binaries or format conversion pipelines.

The project is not trying to win by out-optimizing specialized CUDA, DirectML, or vendor SDK stacks across every workload. It is trying to win by being the most direct, comprehensible, and portable way to run local language models in the .NET ecosystem.

## Problem Statement

Developers building local AI features in .NET face repeated friction:

- `llama.cpp` bindings expose a native engine through wrappers, which adds packaging and compatibility overhead.
- ONNX-based runtimes require model conversion and often demand vendor-specific setup before acceleration is usable.
- Hardware support is fragmented across CPU, GPU, and NPU paths, with inconsistent developer stories across platforms.
- High-level .NET abstractions often exist, but the underlying inference stack still feels foreign to ordinary .NET application development.

This creates a gap between model availability and practical application development.

## Product Thesis

`llmdot` should treat GGUF as the primary model distribution format for local inference in .NET. The community already distributes a large catalog of quantized models in GGUF, and developers should not have to translate them into a different ecosystem before use. Initial releases will target a specific model family and quantization set, expanding coverage as the runtime matures.

The product thesis is:

- GGUF is the model ingestion format
- managed .NET is the orchestration layer
- CPU is the default execution target
- optional accelerators are additive, not foundational

This keeps the project accessible while preserving a path to better performance over time.

## Target Users

The initial audience is:

- .NET application developers building desktop, server, and local tooling workflows
- developers who want private or offline inference without operational complexity
- teams using `Microsoft.Extensions.AI` and related abstractions
- developers targeting small and mid-sized quantized models on consumer hardware

## Value Proposition

`llmdot` should provide a strong answer to a simple developer question:

"How do I run a local GGUF model in .NET without dragging in a native toolchain or converting formats?"

Its value comes from:

- direct model compatibility
- predictable packaging
- idiomatic APIs
- low-friction deployment

## Positioning

### Compared with native llama.cpp wrappers

`llmdot` should be easier to package, easier to host, and easier to reason about inside normal .NET applications. It sacrifices some raw performance headroom in exchange for simpler deployment and a more idiomatic developer experience.

### Compared with ONNX-based runtimes

`llmdot` should be easier to start with, easier to distribute, and more naturally aligned with the models developers are already downloading. It is not expected to beat vendor-optimized ONNX stacks on peak hardware throughput.

## Design Principles

- Default to zero native dependencies in the core path
- Prefer standard .NET abstractions over custom framework surfaces
- Keep model compatibility decoupled from hardware backend choice
- Optimize for the common case of local quantized inference
- Build incremental acceleration paths instead of rewriting the engine around one vendor API

## Success Criteria

The project is succeeding if developers can:

- install a single core package
- load a GGUF model directly
- generate text with streaming responses
- integrate the runtime into standard .NET hosting patterns
- deploy the core runtime without managing platform-specific inference binaries
