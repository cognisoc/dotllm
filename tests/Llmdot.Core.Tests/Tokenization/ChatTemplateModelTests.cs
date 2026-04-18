using Llmdot.Loading;
using Llmdot.Tokenization;
using Llmdot.Tokenization.Jinja;
using Xunit;

namespace Llmdot.Core.Tests.Tokenization;

public class ChatTemplateModelTests
{
    private static ChatTemplate CreateTemplate(string template)
    {
        var tokens = new[] { "<pad>", "<s>", "</s>" };
        var scores = new float[] { 0f, 0f, 0f };
        var merges = Array.Empty<string>();
        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 1, eosTokenId: 2);

        return ChatTemplate.FromGguf(
            new GgufMetadata(new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.chat_template"] = new GgufMetadataValue { Type = GgufValueType.String, Value = template },
            }),
            tokenizer)!;
    }

    [Fact]
    public void ChatML_Generic_WithGenerationPrompt()
    {
        var template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        var ct = CreateTemplate(template);

        var result = ct.Format([
            new ChatMessageEntry("user", "Hello"),
        ]);

        Assert.Contains("<|im_start|>user\nHello<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void ChatML_SystemMessage()
    {
        var template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        var ct = CreateTemplate(template);

        var result = ct.Format([
            new ChatMessageEntry("system", "You are helpful."),
            new ChatMessageEntry("user", "Hello"),
        ]);

        Assert.Contains("<|im_start|>system\nYou are helpful.<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>", result);
    }

    [Fact]
    public void ChatML_MultiTurn()
    {
        var template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        var ct = CreateTemplate(template);

        var result = ct.Format([
            new ChatMessageEntry("user", "Hi"),
            new ChatMessageEntry("assistant", "Hello!"),
            new ChatMessageEntry("user", "How are you?"),
        ]);

        Assert.Contains("<|im_start|>user\nHi<|im_end|>", result);
        Assert.Contains("<|im_start|>assistant\nHello!<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nHow are you?<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void Template_WithMapFilter()
    {
        var template = "Roles: {{ messages|map('role')|join(', ') }}";
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?> { ["role"] = "user", ["content"] = "hi" },
                new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "hello" },
            },
        };

        var result = JinjaTemplate.Render(template, ctx);
        Assert.Equal("Roles: user, assistant", result);
    }

    [Fact]
    public void Template_WithSelectAttr()
    {
        var template = "{% for msg in messages|selectattr('role', 'equalto', 'user') %}{{ msg.content }}{% endfor %}";
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?> { ["role"] = "system", ["content"] = "sys" },
                new Dictionary<string, object?> { ["role"] = "user", ["content"] = "hello" },
                new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "hi" },
            },
        };

        var result = JinjaTemplate.Render(template, ctx);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Template_WithRaiseException()
    {
        var template = "{% if role == 'unknown' %}{{ raise_exception('Unsupported role') }}{% endif %}ok";
        var ctx = new Dictionary<string, object?>
        {
            ["role"] = "unknown",
            ["raise_exception"] = (Func<object?[], Dictionary<string, object?>, object?>)((args, _) =>
                throw new InvalidOperationException(args.Length > 0 ? args[0]?.ToString() ?? "error" : "error")),
        };

        Assert.Throws<InvalidOperationException>(() => JinjaTemplate.Render(template, ctx));
    }
}
