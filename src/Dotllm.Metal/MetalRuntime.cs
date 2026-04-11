using System.Runtime.InteropServices;

namespace Dotllm.Metal;

internal static class MetalRuntime
{
    private static bool? _isAvailable;

    public static bool IsAvailable
    {
        get
        {
            if (_isAvailable.HasValue) return _isAvailable.Value;

            _isAvailable = false;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                try
                {
                    _isAvailable = NativeLibrary.TryLoad("/System/Library/Frameworks/Metal.framework/Metal", out _);
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