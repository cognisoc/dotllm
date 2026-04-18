using System.Runtime.InteropServices;
using Llmdot.Vulkan.Native;

namespace Llmdot.Vulkan.Native;

internal sealed unsafe class VulkanInterop : IDisposable
{
    private nint _instance;
    private nint _physicalDevice;
    private nint _device;
    private nint _queue;
    private nint _commandPool;
    private nint _descriptorPool;
    private nint _pipelineLayout;
    private nint _descriptorSetLayout;
    private uint _computeQueueFamilyIndex;
    private bool _disposed;

    public nint Device => _device;
    public nint PhysicalDevice => _physicalDevice;
    public nint Queue => _queue;
    public nint CommandPool => _commandPool;
    public nint DescriptorPool => _descriptorPool;
    public nint PipelineLayout => _pipelineLayout;
    public nint DescriptorSetLayout => _descriptorSetLayout;
    public bool IsInitialized => _device != 0;

    public VulkanInterop(VulkanBackendOptions? options = null)
    {
        if (!VulkanRuntime.IsAvailable)
            throw new InvalidOperationException("Vulkan runtime not available");

        CreateInstance();
        SelectPhysicalDevice(options?.DeviceIndex ?? 0);
        CreateDevice();
        CreateCommandPool();
        CreateDescriptorPool();
        CreatePipelineLayout();
    }

    private void CreateInstance()
    {
        var appInfo = new VkApplicationInfo
        {
            sType = VulkanNative.STYPE_APPLICATION_INFO,
            apiVersion = (uint)((4 << 22) | (3 << 12))
        };

        var createInfo = new VkInstanceCreateInfo
        {
            sType = VulkanNative.STYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo = (nint)(&appInfo)
        };

        nint instance;
        var result = VulkanNative.vkCreateInstance(&createInfo, 0, &instance);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException($"Failed to create Vulkan instance: {result}");

        _instance = instance;
    }

    private void SelectPhysicalDevice(int deviceIndex)
    {
        uint count = 0;
        var enumResult = VulkanNative.vkEnumeratePhysicalDevices(_instance, &count, null);
        if (enumResult != VulkanNative.VK_SUCCESS || count == 0)
            throw new InvalidOperationException("No Vulkan physical devices found");

        var devices = stackalloc nint[(int)count];
        _ = VulkanNative.vkEnumeratePhysicalDevices(_instance, &count, devices);

        _physicalDevice = deviceIndex < count ? devices[deviceIndex] : devices[0];

        uint queueCount = 0;
        VulkanNative.vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueCount, 0);
        var queueProps = stackalloc byte[(int)(queueCount * 64)];
        VulkanNative.vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &queueCount, (nint)queueProps);

        _computeQueueFamilyIndex = 0;
    }

    private void CreateDevice()
    {
        float priority = 1.0f;
        var queueCreateInfo = new VkDeviceQueueCreateInfo
        {
            sType = VulkanNative.STYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex = _computeQueueFamilyIndex,
            queueCount = 1,
            pQueuePriorities = (nint)(&priority)
        };

        var deviceCreateInfo = new VkDeviceCreateInfo
        {
            sType = VulkanNative.STYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount = 1,
            pQueueCreateInfos = (nint)(&queueCreateInfo)
        };

        nint device;
        var result = VulkanNative.vkCreateDevice(_physicalDevice, &deviceCreateInfo, 0, &device);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException($"Failed to create Vulkan device: {result}");

        _device = device;

        nint queue;
        VulkanNative.vkGetDeviceQueue(_device, _computeQueueFamilyIndex, 0, &queue);
        _queue = queue;
    }

    private void CreateCommandPool()
    {
        var createInfo = new VkCommandPoolCreateInfo
        {
            sType = VulkanNative.STYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex = _computeQueueFamilyIndex
        };

        nint pool;
        var result = VulkanNative.vkCreateCommandPool(_device, &createInfo, 0, &pool);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException("Failed to create Vulkan command pool");

        _commandPool = pool;
    }

    private void CreateDescriptorPool()
    {
        var poolSizes = stackalloc VkDescriptorPoolSize[2];
        poolSizes[0] = new VkDescriptorPoolSize
        {
            type = VulkanNative.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount = 1024
        };
        poolSizes[1] = new VkDescriptorPoolSize
        {
            type = VulkanNative.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount = 1024
        };

        var createInfo = new VkDescriptorPoolCreateInfo
        {
            sType = VulkanNative.STYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets = 1024,
            poolSizeCount = 2,
            pPoolSizes = (nint)poolSizes
        };

        nint pool;
        var result = VulkanNative.vkCreateDescriptorPool(_device, &createInfo, 0, &pool);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException("Failed to create Vulkan descriptor pool");

        _descriptorPool = pool;
    }

    private void CreateDescriptorSetLayout(uint bindingCount)
    {
        var bindings = stackalloc VkDescriptorSetLayoutBinding[(int)bindingCount];
        for (uint i = 0; i < bindingCount; i++)
        {
            bindings[i] = new VkDescriptorSetLayoutBinding
            {
                binding = i,
                descriptorType = VulkanNative.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount = 1,
                stageFlags = VulkanNative.VK_SHADER_STAGE_COMPUTE_BIT
            };
        }

        var createInfo = new VkDescriptorSetLayoutCreateInfo
        {
            sType = VulkanNative.STYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = bindingCount,
            pBindings = (nint)bindings
        };

        nint layout;
        var result = VulkanNative.vkCreateDescriptorSetLayout(_device, &createInfo, 0, &layout);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException("Failed to create descriptor set layout");

        _descriptorSetLayout = layout;
    }

    private void CreatePipelineLayout()
    {
        CreateDescriptorSetLayout(8);

        var layout = _descriptorSetLayout;
        var pushConstantRange = new VkPushConstantRange
        {
            stageFlags = VulkanNative.VK_SHADER_STAGE_COMPUTE_BIT,
            offset = 0,
            size = 64
        };

        var createInfo = new VkPipelineLayoutCreateInfo
        {
            sType = VulkanNative.STYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = (nint)(&layout),
            pushConstantRangeCount = 1,
            pPushConstantRanges = (nint)(&pushConstantRange)
        };

        nint pipelineLayout;
        var result = VulkanNative.vkCreatePipelineLayout(_device, &createInfo, 0, &pipelineLayout);
        if (result != VulkanNative.VK_SUCCESS)
            throw new InvalidOperationException("Failed to create pipeline layout");

        _pipelineLayout = pipelineLayout;
    }

    public nint CreateBuffer(ulong size, uint usage)
    {
        var createInfo = new VkBufferCreateInfo
        {
            sType = VulkanNative.STYPE_BUFFER_CREATE_INFO,
            size = size,
            usage = usage,
            sharingMode = VulkanNative.VK_SHARING_MODE_EXCLUSIVE
        };

        nint buffer;
        var result = VulkanNative.vkCreateBuffer(_device, &createInfo, 0, &buffer);
        return result == VulkanNative.VK_SUCCESS ? buffer : 0;
    }

    public nint AllocateBufferMemory(nint buffer)
    {
        VkMemoryRequirements reqs;
        VulkanNative.vkGetBufferMemoryRequirements(_device, buffer, &reqs);

        VkPhysicalDeviceMemoryProperties memProps;
        VulkanNative.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProps);

        uint memoryTypeIndex = FindMemoryType(reqs.memoryTypeBits,
            VulkanNative.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanNative.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &memProps);

        var allocInfo = new VkMemoryAllocateInfo
        {
            sType = VulkanNative.STYPE_MEMORY_ALLOCATE_INFO,
            allocationSize = reqs.size,
            memoryTypeIndex = memoryTypeIndex
        };

        nint memory;
        var result = VulkanNative.vkAllocateMemory(_device, &allocInfo, 0, &memory);
        if (result != VulkanNative.VK_SUCCESS)
            return 0;

        _ = VulkanNative.vkBindBufferMemory(_device, buffer, memory, 0);
        return memory;
    }

    private uint FindMemoryType(uint typeBits, uint properties, VkPhysicalDeviceMemoryProperties* memProps)
    {
        for (uint i = 0; i < memProps->memoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) != 0)
            {
                var memType = (VkMemoryType*)((nint)memProps + sizeof(uint) + i * sizeof(VkMemoryType));
                if ((memType->propertyFlags & properties) == properties)
                    return i;
            }
        }
        return 0;
    }

    public nint MapMemory(nint memory, ulong size)
    {
        nint data;
        _ = VulkanNative.vkMapMemory(_device, memory, 0, size, 0, &data);
        return data;
    }

    public void UnmapMemory(nint memory) => VulkanNative.vkUnmapMemory(_device, memory);
    public void FreeMemory(nint memory) { if (memory != 0) VulkanNative.vkFreeMemory(_device, memory, 0); }
    public void DestroyBuffer(nint buffer) { if (buffer != 0) VulkanNative.vkDestroyBuffer(_device, buffer, 0); }

    public nint CreateComputePipeline(nint shaderModule, string entryPoint)
    {
        fixed (byte* namePtr = System.Text.Encoding.UTF8.GetBytes(entryPoint + "\0"))
        {
            var stageCreateInfo = new VkPipelineShaderStageCreateInfo
            {
                sType = VulkanNative.STYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = VulkanNative.VK_SHADER_STAGE_COMPUTE_BIT,
                module = shaderModule,
                pName = (nint)namePtr
            };

            var pipelineCreateInfo = new VkComputePipelineCreateInfo
            {
                sType = VulkanNative.STYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage = stageCreateInfo,
                layout = _pipelineLayout
            };

            nint pipeline;
            var result = VulkanNative.vkCreateComputePipelines(_device, 0, 1, &pipelineCreateInfo, 0, &pipeline);
            return result == VulkanNative.VK_SUCCESS ? pipeline : 0;
        }
    }

    public nint CreateShaderModule(ReadOnlySpan<byte> spirvCode)
    {
        fixed (byte* ptr = spirvCode)
        {
            var createInfo = new VkShaderModuleCreateInfo
            {
                sType = VulkanNative.STYPE_SHADER_MODULE_CREATE_INFO,
                codeSize = (nuint)spirvCode.Length,
                pCode = (nint)ptr
            };

            nint module;
            var result = VulkanNative.vkCreateShaderModule(_device, &createInfo, 0, &module);
            return result == VulkanNative.VK_SUCCESS ? module : 0;
        }
    }

    public void DestroyShaderModule(nint module) { if (module != 0) VulkanNative.vkDestroyShaderModule(_device, module, 0); }
    public void DestroyPipeline(nint pipeline) { if (pipeline != 0) VulkanNative.vkDestroyPipeline(_device, pipeline, 0); }

    public static nint AllocateCommandBuffer(nint device, nint commandPool)
    {
        var allocInfo = new VkCommandBufferAllocateInfo
        {
            sType = VulkanNative.STYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool = commandPool,
            level = 0,
            commandBufferCount = 1
        };

        nint cmdBuffer;
        _ = VulkanNative.vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer);
        return cmdBuffer;
    }

    public static void BeginCommandBuffer(nint cmdBuffer)
    {
        var beginInfo = new VkCommandBufferBeginInfo
        {
            sType = VulkanNative.STYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags = VulkanNative.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        _ = VulkanNative.vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    }

    public static void EndCommandBuffer(nint cmdBuffer) { _ = VulkanNative.vkEndCommandBuffer(cmdBuffer); }

    public void SubmitAndWait(nint cmdBuffer)
    {
        nint fence;
        var fenceInfo = new VkFenceCreateInfo
        {
            sType = VulkanNative.STYPE_FENCE_CREATE_INFO
        };
        _ = VulkanNative.vkCreateFence(_device, &fenceInfo, 0, &fence);

        var submitInfo = new VkSubmitInfo
        {
            sType = VulkanNative.STYPE_SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = (nint)(&cmdBuffer)
        };

        _ = VulkanNative.vkQueueSubmit(_queue, 1, &submitInfo, fence);
        _ = VulkanNative.vkWaitForFences(_device, 1, &fence, 1, ulong.MaxValue);
        VulkanNative.vkDestroyFence(_device, fence, 0);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_pipelineLayout != 0) { VulkanNative.vkDestroyPipelineLayout(_device, _pipelineLayout, 0); _pipelineLayout = 0; }
        if (_descriptorSetLayout != 0) { VulkanNative.vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, 0); _descriptorSetLayout = 0; }
        if (_descriptorPool != 0) { VulkanNative.vkDestroyDescriptorPool(_device, _descriptorPool, 0); _descriptorPool = 0; }
        if (_commandPool != 0) { VulkanNative.vkDestroyCommandPool(_device, _commandPool, 0); _commandPool = 0; }
        if (_device != 0) { VulkanNative.vkDestroyDevice(_device, 0); _device = 0; }
        if (_instance != 0) { VulkanNative.vkDestroyInstance(_instance, 0); _instance = 0; }
    }
}