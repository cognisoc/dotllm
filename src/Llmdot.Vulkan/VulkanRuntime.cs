using System.Runtime.InteropServices;

namespace Llmdot.Vulkan;

internal static class VulkanRuntime
{
    private static bool? _isAvailable;

    public static bool IsAvailable
    {
        get
        {
            if (_isAvailable.HasValue) return _isAvailable.Value;

            _isAvailable = false;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
                RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                try
                {
                    _isAvailable = NativeLibrary.TryLoad("vulkan-1", out _) ||
                                   NativeLibrary.TryLoad("libvulkan.so.1", out _) ||
                                   NativeLibrary.TryLoad("libvulkan.so", out _);
                }
                catch
                {
                    _isAvailable = false;
                }
            }

            return _isAvailable.Value;
        }
    }
}