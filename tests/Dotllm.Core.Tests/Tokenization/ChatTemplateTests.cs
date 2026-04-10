using Dotllm.Loading;
using Dotllm.Tokenization;
using Xunit;

namespace Dotllm.Core.Tests.Tokenization;

public class ChatTemplateTests
{
    [Fact]
    public void Format_ChatML_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("system", "You are helpful."),
            new ChatMessageEntry("user", "Hello!"),
        ]);

        Assert.Equal("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n", result);
    }

    [Fact]
    public void Format_Llama3_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|begin_of_text|>", "<|eot_id|>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Contains("<|start_header_id|>user<|end_header_id|>", result);
        Assert.Contains("Hi<|eot_id|>", result);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", result);
    }

    [Fact]
    public void Format_Gemma_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{{ bos_token }}{% for message in messages %}<start_of_turn>{{ message.role }}\n{{ message.content }}<end_of_turn>\n{% endfor %}<start_of_turn>model\n",
            "<bos>", "<eos>");

        var result = template.Format([
            new ChatMessageEntry("user", "What is 2+2?"),
        ]);

        Assert.Equal("<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n", result);
    }

    [Fact]
    public void Format_Phi3_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}<|end|>\n{% endfor %}<|assistant|>\n",
            "<s>", "<|end|>");

        var result = template.Format([
            new ChatMessageEntry("user", "Summarize this."),
        ]);

        Assert.Contains("|>user", result);
        Assert.Contains("Summarize this.", result);
        Assert.Contains("<|end|>", result);
        Assert.Contains("assistant", result);
    }

    [Fact]
    public void Format_Mistral_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{{ bos_token }}{% for message in messages %}{% if message.role == 'user' %}{{ '[INST] ' + message.content + ' [/INST]' }}{% elif message.role == 'assistant' %}{{ message.content + eos_token + ' ' }}{% endif %}{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hello"),
            new ChatMessageEntry("assistant", "Hi there"),
        ]);

        Assert.Contains("[INST] Hello [/INST]", result);
        Assert.Contains("Hi there</s>", result);
    }

    [Fact]
    public void Format_Qwen2_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("system", "You are helpful."),
            new ChatMessageEntry("user", "Hello"),
        ]);

        Assert.Contains("<|im_start|>system\nYou are helpful.<|im_end|>", result);
        Assert.Contains("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void Format_Llama2System_ProducesCorrectFormat()
    {
        var template = CreateTemplate(
            "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token }}{% if system_message != '' %}[INST] <<SYS>>\n{{ system_message }}\n<</SYS>>\n\n{% endif %}{% for message in messages %}{% if message.role == 'user' %}[INST] {{ message.content }} [/INST]{% elif message.role == 'assistant' %}{{ message.content }}{{ eos_token }}{% endif %}{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("system", "Be helpful."),
            new ChatMessageEntry("user", "Explain quantum computing."),
        ]);

        Assert.Contains("<<SYS>>", result);
        Assert.Contains("Be helpful.", result);
        Assert.Contains("[INST] Explain quantum computing. [/INST]", result);
    }

    [Fact]
    public void Format_WithSetAndReassignment_WorksCorrectly()
    {
        var template = CreateTemplate(
            "{% set ns = messages %}{% for message in ns %}{{ message.role }}: {{ message.content }}\n{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
            new ChatMessageEntry("assistant", "Hello"),
        ]);

        Assert.Equal("user: Hi\nassistant: Hello\n", result);
    }

    [Fact]
    public void Format_LoopIndex_VariablesAreCorrect()
    {
        var template = CreateTemplate(
            "{% for message in messages %}{{ loop.index0 }}: {{ message.role }}\n{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "A"),
            new ChatMessageEntry("assistant", "B"),
            new ChatMessageEntry("user", "C"),
        ]);

        Assert.Equal("0: user\n1: assistant\n2: user\n", result);
    }

    [Fact]
    public void Format_WhitespaceTrimDashLeft()
    {
        var template = CreateTemplate(
            "hello   {%- for message in messages %}{{ message.role }}{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("hellouser", result);
    }

    [Fact]
    public void Format_WhitespaceTrimDashRight()
    {
        var template = CreateTemplate(
            "{%- for message in messages -%} {{ message.role }}{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("user", result);
    }

    [Fact]
    public void Format_CommentsAreRemoved()
    {
        var template = CreateTemplate(
            "{# this is a comment #}before{# another #}after",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("beforeafter", result);
    }

    [Fact]
    public void Format_FilterLengthOnMessages()
    {
        var template = CreateTemplate(
            "{% if messages|length > 0 %}has-messages{% endif %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("has-messages", result);
    }

    [Fact]
    public void Format_BosEosTokensAreAvailable()
    {
        var template = CreateTemplate(
            "{{ bos_token }}{{ eos_token }}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("<s></s>", result);
    }

    [Fact]
    public void Format_IsDefinedCheck()
    {
        var template = CreateTemplate(
            "{% if tools is defined %}tools{% else %}no-tools{% endif %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("no-tools", result);
    }

    [Fact]
    public void Format_AddGenerationPromptIsTrue()
    {
        var template = CreateTemplate(
            "{% if add_generation_prompt %}generate{% endif %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("generate", result);
    }

    [Fact]
    public void Format_DictAndBracketAccess()
    {
        var template = CreateTemplate(
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}",
            "<s>", "</s>");

        var result = template.Format([
            new ChatMessageEntry("user", "Hi"),
        ]);

        Assert.Equal("user: Hi\n", result);
    }

    private static ChatTemplate CreateTemplate(string rawTemplate, string bosToken, string eosToken)
    {
        var tokens = new[] { "<pad>", bosToken, eosToken };
        var scores = new float[] { 0f, 0f, 0f };
        var merges = Array.Empty<string>();

        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 1, eosTokenId: 2);

        return ChatTemplate.FromGguf(
            new GgufMetadata(new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.chat_template"] = new GgufMetadataValue { Type = GgufValueType.String, Value = rawTemplate },
            }),
            tokenizer)!;
    }
}