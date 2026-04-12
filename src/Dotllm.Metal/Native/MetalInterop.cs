using System.Runtime.InteropServices;

namespace Dotllm.Metal.Native;

internal sealed unsafe class MetalInterop : IDisposable
{
    private nint _device;
    private nint _commandQueue;
    private bool _disposed;

    private static readonly nint SelNewCommandQueue = ObjC.Sel("newCommandQueue");
    private static readonly nint SelNewBufferLength = ObjC.Sel("newBufferWithLength:options:");
    private static readonly nint SelNewBufferBytes = ObjC.Sel("newBufferWithBytes:length:options:");
    private static readonly nint SelContents = ObjC.Sel("contents");
    private static readonly nint SelDidModifyRange = ObjC.Sel("didModifyRange:");
    private static readonly nint SelNewLibrarySource = ObjC.Sel("newLibraryWithSource:options:error:");
    private static readonly nint SelNewFunction = ObjC.Sel("newFunctionWithName:");
    private static readonly nint SelNewPipeline = ObjC.Sel("newComputePipelineStateWithFunction:error:");
    private static readonly nint SelCommandBuffer = ObjC.Sel("commandBuffer");
    private static readonly nint SelComputeEncoder = ObjC.Sel("computeCommandEncoder");
    private static readonly nint SelSetPipeline = ObjC.Sel("setComputePipelineState:");
    private static readonly nint SelSetBuffer = ObjC.Sel("setBuffer:offset:atIndex:");
    private static readonly nint SelSetBytes = ObjC.Sel("setBytes:length:atIndex:");
    private static readonly nint SelEndEncoding = ObjC.Sel("endEncoding");
    private static readonly nint SelCommit = ObjC.Sel("commit");
    private static readonly nint SelWaitCompleted = ObjC.Sel("waitUntilCompleted");
    private static readonly nint SelRelease = ObjC.Sel("release");

    [DllImport("/System/Library/Frameworks/Metal.framework/Metal", EntryPoint = "MTLCreateSystemDefaultDevice")]
    private static extern nint MTLCreateSystemDefaultDevice();

    [DllImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static extern void MsgSendDispatch(nint receiver, nint selector,
        ulong gridW, ulong gridH, ulong gridD,
        ulong tgW, ulong tgH, ulong tgD);

    [DllImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    private static extern nint MsgSendStret(nint receiver, nint selector, nint arg1, nint arg2);

    public bool IsInitialized => _device != 0;

    public MetalInterop()
    {
        _device = MTLCreateSystemDefaultDevice();
        if (_device == 0)
            throw new InvalidOperationException("No Metal device available");

        _commandQueue = ObjC.MsgSend(_device, SelNewCommandQueue);
        if (_commandQueue == 0)
            throw new InvalidOperationException("Failed to create Metal command queue");
    }

    public nint CreateBuffer(ulong length, MtlResourceOptions options = MtlResourceOptions.StorageModeShared)
    {
        return ObjC.MsgSend(_device, SelNewBufferLength, (nint)length, (nint)(long)options);
    }

    public nint CreateBufferFromData(ReadOnlySpan<byte> data, MtlResourceOptions options = MtlResourceOptions.StorageModeShared)
    {
        fixed (byte* ptr = data)
        {
            return ObjC.MsgSend(_device, SelNewBufferBytes, (nint)ptr, (nint)data.Length, (nint)(long)options);
        }
    }

    public nint CreateBufferFromFloats(ReadOnlySpan<float> data, MtlResourceOptions options = MtlResourceOptions.StorageModeShared)
    {
        var byteSpan = MemoryMarshal.AsBytes(data);
        fixed (byte* ptr = byteSpan)
        {
            return ObjC.MsgSend(_device, SelNewBufferBytes, (nint)ptr, (nint)byteSpan.Length, (nint)(long)options);
        }
    }

    public void CopyToBuffer(nint buffer, ReadOnlySpan<float> data)
    {
        var contentsPtr = ObjC.MsgSend(buffer, SelContents);
        var byteSpan = MemoryMarshal.AsBytes(data);
        fixed (byte* src = byteSpan)
        {
            Buffer.MemoryCopy(src, (void*)contentsPtr, byteSpan.Length, byteSpan.Length);
        }
        ObjC.MsgSend(buffer, SelDidModifyRange, (nint)0, (nint)byteSpan.Length);
    }

    public void CopyToBufferBytes(nint buffer, ReadOnlySpan<byte> data)
    {
        var contentsPtr = ObjC.MsgSend(buffer, SelContents);
        fixed (byte* src = data)
        {
            Buffer.MemoryCopy(src, (void*)contentsPtr, data.Length, data.Length);
        }
        ObjC.MsgSend(buffer, SelDidModifyRange, (nint)0, (nint)data.Length);
    }

    public void CopyFromBuffer(nint buffer, Span<float> destination)
    {
        var contentsPtr = ObjC.MsgSend(buffer, SelContents);
        var byteSpan = MemoryMarshal.AsBytes(destination);
        fixed (byte* dst = byteSpan)
        {
            Buffer.MemoryCopy((void*)contentsPtr, dst, byteSpan.Length, byteSpan.Length);
        }
    }

    public nint CompileShader(string source, string functionName)
    {
        var nsSource = ObjC.CreateNSString(source);
        var nsFuncName = ObjC.CreateNSString(functionName);

        nint errorPtr = 0;
        var library = ObjC.MsgSend(_device, SelNewLibrarySource, nsSource, 0, (nint)(&errorPtr));
        if (library == 0)
            throw new InvalidOperationException($"Failed to compile Metal shader '{functionName}'");

        var function = ObjC.MsgSend(library, SelNewFunction, nsFuncName);
        if (function == 0)
            throw new InvalidOperationException($"Failed to create Metal function '{functionName}'");

        var pipeline = ObjC.MsgSend(_device, SelNewPipeline, function, (nint)(&errorPtr));
        if (pipeline == 0)
            throw new InvalidOperationException($"Failed to create Metal pipeline for '{functionName}'");

        return pipeline;
    }

    public nint CreateCommandBuffer()
    {
        return ObjC.MsgSend(_commandQueue, SelCommandBuffer);
    }

    public static nint CreateComputeEncoder(nint commandBuffer)
    {
        return ObjC.MsgSend(commandBuffer, SelComputeEncoder);
    }

    public static void SetPipeline(nint encoder, nint pipeline)
    {
        ObjC.MsgSend(encoder, SelSetPipeline, pipeline);
    }

    public static void SetBuffer(nint encoder, nint buffer, uint offset, uint index)
    {
        ObjC.MsgSend(encoder, SelSetBuffer, buffer, (nint)offset, (nint)index);
    }

    public static void SetBytesFloat(nint encoder, float value, uint index)
    {
        ObjC.MsgSend(encoder, SelSetBytes, (nint)(&value), (nint)sizeof(float), (nint)index);
    }

    public static void SetBytesUint(nint encoder, uint value, uint index)
    {
        ObjC.MsgSend(encoder, SelSetBytes, (nint)(&value), (nint)sizeof(uint), (nint)index);
    }

    public static void Dispatch1D(nint encoder, uint totalThreads, uint threadsPerGroup)
    {
        var threadgroups = (totalThreads + threadsPerGroup - 1) / threadsPerGroup;
        MsgSendDispatch(encoder, ObjC.Sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            threadgroups, 1, 1, threadsPerGroup, 1, 1);
    }

    public static void EndEncoding(nint encoder)
    {
        ObjC.MsgSend(encoder, SelEndEncoding);
    }

    public static void Commit(nint commandBuffer)
    {
        ObjC.MsgSend(commandBuffer, SelCommit);
    }

    public static void WaitUntilCompleted(nint commandBuffer)
    {
        ObjC.MsgSend(commandBuffer, SelWaitCompleted);
    }

    public static void DestroyBuffer(nint buffer)
    {
        if (buffer != 0)
            ObjC.MsgSend(buffer, SelRelease);
    }

    public static void DestroyPipeline(nint pipeline)
    {
        if (pipeline != 0)
            ObjC.MsgSend(pipeline, SelRelease);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_commandQueue != 0)
        {
            ObjC.MsgSend(_commandQueue, SelRelease);
            _commandQueue = 0;
        }

        if (_device != 0)
        {
            ObjC.MsgSend(_device, SelRelease);
            _device = 0;
        }
    }
}