namespace Dotllm.Vulkan;

public sealed class VulkanBackendOptions
{
    public bool EnableValidationLayers { get; set; }
    public int DeviceIndex { get; set; } = 0;
}