using Dotllm.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.AI;
using Xunit;

namespace Dotllm.Extensions.AI.Tests;

public class DotllmOptionsTests
{
    [Fact]
    public void Defaults_AreSet()
    {
        var options = new DotllmOptions();

        Assert.Equal(string.Empty, options.ModelPath);
        Assert.Equal(4096, options.ContextLength);
        Assert.Equal(256, options.MaxTokens);
        Assert.Equal(0.8f, options.Temperature);
        Assert.Equal(40, options.TopK);
        Assert.Equal(0.95f, options.TopP);
        Assert.Equal(1.1f, options.RepeatPenalty);
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var options = new DotllmOptions
        {
            ModelPath = "/path/to/model.gguf",
            ContextLength = 8192,
            MaxTokens = 512,
            Temperature = 0.5f,
            TopK = 80,
            TopP = 0.9f,
            RepeatPenalty = 1.2f,
        };

        Assert.Equal("/path/to/model.gguf", options.ModelPath);
        Assert.Equal(8192, options.ContextLength);
        Assert.Equal(512, options.MaxTokens);
        Assert.Equal(0.5f, options.Temperature);
        Assert.Equal(80, options.TopK);
        Assert.Equal(0.9f, options.TopP);
        Assert.Equal(1.2f, options.RepeatPenalty);
    }
}

public class DotllmServiceCollectionExtensionsTests
{
    [Fact]
    public void AddDotllm_WithPath_RegistersDotllmChatClientAndIChatClient()
    {
        var services = new ServiceCollection();
        services.AddDotllm("/path/to/model.gguf");

        Assert.Contains(services, sd => sd.ServiceType == typeof(DotllmChatClient));
        Assert.Contains(services, sd => sd.ServiceType == typeof(IChatClient));
    }

    [Fact]
    public void AddDotllm_WithAction_RegistersDotllmChatClientAndIChatClient()
    {
        var services = new ServiceCollection();
        services.AddDotllm(options => options.ModelPath = "/path/to/model.gguf");

        Assert.Contains(services, sd => sd.ServiceType == typeof(DotllmChatClient));
        Assert.Contains(services, sd => sd.ServiceType == typeof(IChatClient));
    }
}

public class DotllmChatClientTests
{
    [Fact]
    public void Constructor_WithEmptyModelPath_Throws()
    {
        var options = new DotllmOptions();
        var ioptions = Microsoft.Extensions.Options.Options.Create(options);

        Assert.Throws<InvalidOperationException>(() =>
        {
            using var client = new DotllmChatClient(ioptions);
        });
    }
}