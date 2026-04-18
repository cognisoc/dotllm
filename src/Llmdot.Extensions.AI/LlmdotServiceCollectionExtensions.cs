using Llmdot.Inference;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace Llmdot.Extensions.AI;

public static class LlmdotServiceCollectionExtensions
{
    public static IServiceCollection AddLlmdot(this IServiceCollection services, string modelPath)
    {
        return services.AddLlmdot(options => options.ModelPath = modelPath);
    }

    public static IServiceCollection AddLlmdot(this IServiceCollection services, Action<LlmdotOptions> configure)
    {
        services.Configure(configure);
        services.AddSingleton<LlmdotChatClient>();
        services.AddSingleton<IChatClient>(sp => sp.GetRequiredService<LlmdotChatClient>());
        return services;
    }
}