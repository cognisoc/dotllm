using System.Runtime.InteropServices;
using Dotllm.Inference;

namespace Dotllm.Extensions.AI;

public static class BackendFactory
{
    public static IComputeBackend CreateBestAvailable()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            try
            {
                if (NativeLibrary.TryLoad("/System/Library/Frameworks/Metal.framework/Metal", out _))
                    return new Dotllm.Metal.MetalBackend();
            }
            catch
            {
                
            }
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var libName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "vulkan-1" : "libvulkan.so.1";
                if (NativeLibrary.TryLoad(libName, out _))
                    return new Dotllm.Vulkan.VulkanBackend();
            }
            catch
            {
                
            }
        }

        return new CpuBackend();
    }

    public static IComputeBackend CreateCpu() => new CpuBackend();

    public static (IComputeBackend Backend, string Name) CreateBestAvailableWithInfo()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            try
            {
                if (NativeLibrary.TryLoad("/System/Library/Frameworks/Metal.framework/Metal", out _))
                    return (new Dotllm.Metal.MetalBackend(), "Metal");
            }
            catch { }
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            try
            {
                var libName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "vulkan-1" : "libvulkan.so.1";
                if (NativeLibrary.TryLoad(libName, out _))
                    return (new Dotllm.Vulkan.VulkanBackend(), "Vulkan");
            }
            catch { }
        }

        return (new CpuBackend(), "CPU");
    }
}