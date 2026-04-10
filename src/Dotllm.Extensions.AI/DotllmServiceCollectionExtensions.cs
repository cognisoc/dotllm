using Dotllm.Inference;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace Dotllm.Extensions.AI;

public static class DotllmServiceCollectionExtensions
{
    public static IServiceCollection AddDotllm(this IServiceCollection services, string modelPath)
    {
        return services.AddDotllm(options => options.ModelPath = modelPath);
    }

    public static IServiceCollection AddDotllm(this IServiceCollection services, Action<DotllmOptions> configure)
    {
        services.Configure(configure);
        services.AddSingleton<DotllmChatClient>();
        services.AddSingleton<IChatClient>(sp => sp.GetRequiredService<DotllmChatClient>());
        return services;
    }
}