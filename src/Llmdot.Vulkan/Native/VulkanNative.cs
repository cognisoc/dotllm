using System.Runtime.InteropServices;

namespace Llmdot.Vulkan.Native;

internal static unsafe class VulkanNative
{
#if WINDOWS
    private const string Lib = "vulkan-1.dll";
#else
    private const string Lib = "libvulkan.so.1";
#endif

    public const uint VK_SUCCESS = 0;
    public const int VK_ERROR_INITIALIZATION_FAILED = -2;
    public const uint VK_QUEUE_COMPUTE_BIT = 0x00000002;
    public const uint VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
    public const uint VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
    public const uint VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
    public const uint VK_SHARING_MODE_EXCLUSIVE = 0;
    public const uint VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
    public const uint VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
    public const uint VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
    public const uint VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
    public const uint VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
    public const uint VK_PIPELINE_BIND_POINT_COMPUTE = 1;

    public static nint VK_NULL_HANDLE => 0;

    public static nint STYPE_APPLICATION_INFO => (nint)0;
    public static nint STYPE_INSTANCE_CREATE_INFO => (nint)1;
    public static nint STYPE_DEVICE_QUEUE_CREATE_INFO => (nint)2;
    public static nint STYPE_DEVICE_CREATE_INFO => (nint)3;
    public static nint STYPE_SHADER_MODULE_CREATE_INFO => (nint)4;
    public static nint STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO => (nint)5;
    public static nint STYPE_COMPUTE_PIPELINE_CREATE_INFO => (nint)6;
    public static nint STYPE_PIPELINE_LAYOUT_CREATE_INFO => (nint)7;
    public static nint STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO => (nint)8;
    public static nint STYPE_DESCRIPTOR_POOL_CREATE_INFO => (nint)9;
    public static nint STYPE_DESCRIPTOR_SET_ALLOCATE_INFO => (nint)10;
    public static nint STYPE_COMMAND_POOL_CREATE_INFO => (nint)12;
    public static nint STYPE_COMMAND_BUFFER_ALLOCATE_INFO => (nint)13;
    public static nint STYPE_COMMAND_BUFFER_BEGIN_INFO => (nint)14;
    public static nint STYPE_SUBMIT_INFO => (nint)15;
    public static nint STYPE_FENCE_CREATE_INFO => (nint)16;
    public static nint STYPE_BUFFER_CREATE_INFO => (nint)19;
    public static nint STYPE_MEMORY_ALLOCATE_INFO => (nint)20;
    public static nint STYPE_WRITE_DESCRIPTOR_SET => (nint)24;

    [DllImport(Lib, CharSet = CharSet.Ansi)]
    public static extern uint vkCreateInstance(VkInstanceCreateInfo* pCreateInfo, nint pAllocator, nint* pInstance);

    [DllImport(Lib)]
    public static extern void vkDestroyInstance(nint instance, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkEnumeratePhysicalDevices(nint instance, uint* pPhysicalDeviceCount, nint* pPhysicalDevices);

    [DllImport(Lib)]
    public static extern void vkGetPhysicalDeviceProperties(nint physicalDevice, nint pProperties);

    [DllImport(Lib)]
    public static extern void vkGetPhysicalDeviceMemoryProperties(nint physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties);

    [DllImport(Lib)]
    public static extern void vkGetPhysicalDeviceQueueFamilyProperties(nint physicalDevice, uint* pQueueFamilyPropertyCount, nint pQueueFamilyProperties);

    [DllImport(Lib)]
    public static extern uint vkCreateDevice(nint physicalDevice, VkDeviceCreateInfo* pCreateInfo, nint pAllocator, nint* pDevice);

    [DllImport(Lib)]
    public static extern void vkDestroyDevice(nint device, nint pAllocator);

    [DllImport(Lib)]
    public static extern void vkGetDeviceQueue(nint device, uint queueFamilyIndex, uint queueIndex, nint* pQueue);

    [DllImport(Lib)]
    public static extern uint vkCreateShaderModule(nint device, VkShaderModuleCreateInfo* pCreateInfo, nint pAllocator, nint* pShaderModule);

    [DllImport(Lib)]
    public static extern void vkDestroyShaderModule(nint device, nint shaderModule, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkCreateComputePipelines(nint device, nint pipelineCache, uint createInfoCount, VkComputePipelineCreateInfo* pCreateInfos, nint pAllocator, nint* pPipelines);

    [DllImport(Lib)]
    public static extern void vkDestroyPipeline(nint device, nint pipeline, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkCreatePipelineLayout(nint device, VkPipelineLayoutCreateInfo* pCreateInfo, nint pAllocator, nint* pPipelineLayout);

    [DllImport(Lib)]
    public static extern void vkDestroyPipelineLayout(nint device, nint pipelineLayout, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkCreateDescriptorSetLayout(nint device, VkDescriptorSetLayoutCreateInfo* pCreateInfo, nint pAllocator, nint* pDescriptorSetLayout);

    [DllImport(Lib)]
    public static extern void vkDestroyDescriptorSetLayout(nint device, nint descriptorSetLayout, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkCreateDescriptorPool(nint device, VkDescriptorPoolCreateInfo* pCreateInfo, nint pAllocator, nint* pDescriptorPool);

    [DllImport(Lib)]
    public static extern void vkDestroyDescriptorPool(nint device, nint descriptorPool, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkAllocateDescriptorSets(nint device, VkDescriptorSetAllocateInfo* pAllocateInfo, nint* pDescriptorSets);

    [DllImport(Lib)]
    public static extern void vkUpdateDescriptorSets(nint device, uint descriptorWriteCount, VkWriteDescriptorSet* pDescriptorWrites, uint descriptorCopyCount, nint pDescriptorCopies);

    [DllImport(Lib)]
    public static extern uint vkCreateBuffer(nint device, VkBufferCreateInfo* pCreateInfo, nint pAllocator, nint* pBuffer);

    [DllImport(Lib)]
    public static extern void vkDestroyBuffer(nint device, nint buffer, nint pAllocator);

    [DllImport(Lib)]
    public static extern void vkGetBufferMemoryRequirements(nint device, nint buffer, VkMemoryRequirements* pMemoryRequirements);

    [DllImport(Lib)]
    public static extern uint vkAllocateMemory(nint device, VkMemoryAllocateInfo* pAllocateInfo, nint pAllocator, nint* pMemory);

    [DllImport(Lib)]
    public static extern void vkFreeMemory(nint device, nint memory, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkBindBufferMemory(nint device, nint buffer, nint memory, ulong offset);

    [DllImport(Lib)]
    public static extern uint vkMapMemory(nint device, nint memory, ulong offset, ulong size, uint flags, nint* ppData);

    [DllImport(Lib)]
    public static extern void vkUnmapMemory(nint device, nint memory);

    [DllImport(Lib)]
    public static extern uint vkCreateCommandPool(nint device, VkCommandPoolCreateInfo* pCreateInfo, nint pAllocator, nint* pCommandPool);

    [DllImport(Lib)]
    public static extern void vkDestroyCommandPool(nint device, nint commandPool, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkAllocateCommandBuffers(nint device, VkCommandBufferAllocateInfo* pAllocateInfo, nint* pCommandBuffers);

    [DllImport(Lib)]
    public static extern uint vkBeginCommandBuffer(nint commandBuffer, VkCommandBufferBeginInfo* pBeginInfo);

    [DllImport(Lib)]
    public static extern uint vkEndCommandBuffer(nint commandBuffer);

    [DllImport(Lib)]
    public static extern void vkCmdBindPipeline(nint commandBuffer, uint pipelineBindPoint, nint pipeline);

    [DllImport(Lib)]
    public static extern void vkCmdBindDescriptorSets(nint commandBuffer, uint pipelineBindPoint, nint layout, uint firstSet, uint descriptorSetCount, nint* pDescriptorSets, uint dynamicOffsetCount, nint pDynamicOffsets);

    [DllImport(Lib)]
    public static extern void vkCmdDispatch(nint commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [DllImport(Lib)]
    public static extern void vkCmdPushConstants(nint commandBuffer, nint layout, uint stageFlags, uint offset, uint size, nint pValues);

    [DllImport(Lib)]
    public static extern uint vkQueueSubmit(nint queue, uint submitCount, VkSubmitInfo* pSubmits, nint fence);

    [DllImport(Lib)]
    public static extern uint vkQueueWaitIdle(nint queue);

    [DllImport(Lib)]
    public static extern uint vkCreateFence(nint device, VkFenceCreateInfo* pCreateInfo, nint pAllocator, nint* pFence);

    [DllImport(Lib)]
    public static extern void vkDestroyFence(nint device, nint fence, nint pAllocator);

    [DllImport(Lib)]
    public static extern uint vkWaitForFences(nint device, uint fenceCount, nint* pFences, uint waitAll, ulong timeout);

    [DllImport(Lib)]
    public static extern uint vkResetFences(nint device, uint fenceCount, nint* pFences);
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkApplicationInfo
{
    public nint sType;
    public nint pNext;
    public nint pApplicationName;
    public uint applicationVersion;
    public nint pEngineName;
    public uint engineVersion;
    public uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkInstanceCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public nint pApplicationInfo;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceQueueCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint queueFamilyIndex;
    public uint queueCount;
    public nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint queueCreateInfoCount;
    public nint pQueueCreateInfos;
    public uint enabledLayerCount;
    public nint ppEnabledLayerNames;
    public uint enabledExtensionCount;
    public nint ppEnabledExtensionNames;
    public nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkShaderModuleCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public nuint codeSize;
    public nint pCode;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineShaderStageCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint stage;
    public nint module;
    public nint pName;
    public nint pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkComputePipelineCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public VkPipelineShaderStageCreateInfo stage;
    public nint layout;
    public nint basePipelineHandle;
    public int basePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    public uint stageFlags;
    public uint offset;
    public uint size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineLayoutCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint setLayoutCount;
    public nint pSetLayouts;
    public uint pushConstantRangeCount;
    public nint pPushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutBinding
{
    public uint binding;
    public uint descriptorType;
    public uint descriptorCount;
    public uint stageFlags;
    public nint pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint bindingCount;
    public nint pBindings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolSize
{
    public uint type;
    public uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint maxSets;
    public uint poolSizeCount;
    public nint pPoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetAllocateInfo
{
    public nint sType;
    public nint pNext;
    public nint descriptorPool;
    public uint descriptorSetCount;
    public nint pSetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorBufferInfo
{
    public nint buffer;
    public ulong offset;
    public ulong range;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkWriteDescriptorSet
{
    public nint sType;
    public nint pNext;
    public nint dstSet;
    public uint dstBinding;
    public uint dstArrayElement;
    public uint descriptorCount;
    public uint descriptorType;
    public nint pImageInfo;
    public nint pBufferInfo;
    public nint pTexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public ulong size;
    public uint usage;
    public uint sharingMode;
    public uint queueFamilyIndexCount;
    public nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    public ulong size;
    public ulong alignment;
    public uint memoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryAllocateInfo
{
    public nint sType;
    public nint pNext;
    public ulong allocationSize;
    public uint memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryType
{
    public uint propertyFlags;
    public uint heapIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryHeap
{
    public ulong size;
    public uint flags;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPhysicalDeviceMemoryProperties
{
    public uint memoryTypeCount;
    public VkMemoryType memoryType0;
    public VkMemoryType memoryType1;
    public VkMemoryType memoryType2;
    public VkMemoryType memoryType3;
    public nint _padding0;
    public nint _padding1;
    public nint _padding2;
    public nint _padding3;
    public uint memoryHeapCount;
    public VkMemoryHeap memoryHeap0;
    public VkMemoryHeap memoryHeap1;
    public nint _heapPad0;
    public nint _heapPad1;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandPoolCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public uint queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferAllocateInfo
{
    public nint sType;
    public nint pNext;
    public nint commandPool;
    public uint level;
    public uint commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferBeginInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
    public nint pInheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkSubmitInfo
{
    public nint sType;
    public nint pNext;
    public uint waitSemaphoreCount;
    public nint pWaitSemaphores;
    public nint pWaitDstStageMask;
    public uint commandBufferCount;
    public nint pCommandBuffers;
    public uint signalSemaphoreCount;
    public nint pSignalSemaphores;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkFenceCreateInfo
{
    public nint sType;
    public nint pNext;
    public uint flags;
}