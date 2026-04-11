using Dotllm.Loading;
using Dotllm.Tokenization;
using Dotllm.Tokenization.Jinja;
using Xunit;

namespace Dotllm.Core.Tests.Tokenization;

public class JinjaTests
{
    [Fact]
    public void Render_PlainText_PassesThrough()
    {
        Assert.Equal("hello world", JinjaTemplate.Render("hello world", []));
    }

    [Fact]
    public void Render_VariableExpression_SubstitutesValue()
    {
        var result = JinjaTemplate.Render("Hello {{ name }}!", new Dictionary<string, object?> { ["name"] = "World" });
        Assert.Equal("Hello World!", result);
    }

    [Fact]
    public void Render_DotAccess_ReadsNestedKey()
    {
        var dict = new Dictionary<string, object?> { ["name"] = "test" };
        var result = JinjaTemplate.Render("{{ msg.name }}", new Dictionary<string, object?> { ["msg"] = dict });
        Assert.Equal("test", result);
    }

    [Fact]
    public void Render_BracketAccess_ReadsKey()
    {
        var dict = new Dictionary<string, object?> { ["key"] = "val" };
        var result = JinjaTemplate.Render("{{ data['key'] }}", new Dictionary<string, object?> { ["data"] = dict });
        Assert.Equal("val", result);
    }

    [Fact]
    public void Render_IfBlock_TrueBranch()
    {
        var result = JinjaTemplate.Render("{% if x %}yes{% endif %}", new Dictionary<string, object?> { ["x"] = true });
        Assert.Equal("yes", result);
    }

    [Fact]
    public void Render_IfBlock_FalseBranch()
    {
        var result = JinjaTemplate.Render("{% if x %}yes{% endif %}", new Dictionary<string, object?> { ["x"] = false });
        Assert.Equal("", result);
    }

    [Fact]
    public void Render_IfElse_ElseBranch()
    {
        var result = JinjaTemplate.Render("{% if x %}yes{% else %}no{% endif %}", new Dictionary<string, object?> { ["x"] = false });
        Assert.Equal("no", result);
    }

    [Fact]
    public void Render_IfElifElse_ElifBranch()
    {
        var result = JinjaTemplate.Render(
            "{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}",
            new Dictionary<string, object?> { ["x"] = 2 });
        Assert.Equal("two", result);
    }

    [Fact]
    public void Render_ForLoop_Iterates()
    {
        var result = JinjaTemplate.Render(
            "{% for item in items %}{{ item }} {% endfor %}",
            new Dictionary<string, object?> { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("a b c ", result);
    }

    [Fact]
    public void Render_ForLoop_LoopIndex()
    {
        var result = JinjaTemplate.Render(
            "{% for item in items %}{{ loop.index }}{% endfor %}",
            new Dictionary<string, object?> { ["items"] = new List<object?> { "x", "y", "z" } });
        Assert.Equal("123", result);
    }

    [Fact]
    public void Render_ForLoop_LoopFirstLast()
    {
        var result = JinjaTemplate.Render(
            "{% for item in items %}{% if loop.first %}[{% elif loop.last %}]{% else %}-{% endif %}{% endfor %}",
            new Dictionary<string, object?> { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("[-]", result);
    }

    [Fact]
    public void Render_Set_AssignsVariable()
    {
        var result = JinjaTemplate.Render(
            "{% set x = 'hello' %}{{ x }}", []);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Render_FilterTrim()
    {
        var result = JinjaTemplate.Render("{{ '  hello  '|trim }}", []);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Render_FilterLength()
    {
        var result = JinjaTemplate.Render("{{ items|length }}", new Dictionary<string, object?>
            { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("3", result);
    }

    [Fact]
    public void Render_FilterJoin()
    {
        var result = JinjaTemplate.Render("{{ items|join(', ') }}", new Dictionary<string, object?>
            { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("a, b, c", result);
    }

    [Fact]
    public void Render_FilterDefault()
    {
        var result = JinjaTemplate.Render("{{ x|default('fallback') }}", []);
        Assert.Equal("fallback", result);
    }

    [Fact]
    public void Render_IsDefined()
    {
        var result = JinjaTemplate.Render("{% if x is defined %}yes{% else %}no{% endif %}", []);
        Assert.Equal("no", result);
    }

    [Fact]
    public void Render_IsNotDefined()
    {
        var result = JinjaTemplate.Render("{% if x is not defined %}yes{% else %}no{% endif %}", []);
        Assert.Equal("yes", result);
    }

    [Fact]
    public void Render_Ternary()
    {
        var result = JinjaTemplate.Render("{{ 'yes' if x else 'no' }}", new Dictionary<string, object?> { ["x"] = true });
        Assert.Equal("yes", result);
    }

    [Fact]
    public void Render_Concatenation()
    {
        var result = JinjaTemplate.Render("{{ 'hello' ~ ' ' ~ 'world' }}", []);
        Assert.Equal("hello world", result);
    }

    [Fact]
    public void Render_ComparisonOperators()
    {
        var ctx = new Dictionary<string, object?> { ["x"] = 5 };
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x == 5 else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x != 3 else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x > 3 else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x < 10 else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x >= 5 else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if x <= 5 else false }}", ctx));
    }

    [Fact]
    public void Render_LogicalOperators()
    {
        var ctx = new Dictionary<string, object?> { ["a"] = true, ["b"] = false };
        Assert.Equal("true", JinjaTemplate.Render("{{ true if a and not b else false }}", ctx));
        Assert.Equal("true", JinjaTemplate.Render("{{ true if a or b else false }}", ctx));
    }

    [Fact]
    public void Render_NegativeIndex()
    {
        var result = JinjaTemplate.Render("{{ items[-1] }}", new Dictionary<string, object?>
            { ["items"] = new List<object?> { "a", "b", "c" } });
        Assert.Equal("c", result);
    }

    [Fact]
    public void Render_NoneValue()
    {
        var result = JinjaTemplate.Render("{% if x is none %}null{% endif %}", new Dictionary<string, object?> { ["x"] = null });
        Assert.Equal("null", result);
    }

    [Fact]
    public void Render_WhitespaceTrim_DashLeft()
    {
        var result = JinjaTemplate.Render("  {%- set x = 1 %}hello", []);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Render_WhitespaceTrim_DashRight()
    {
        var result = JinjaTemplate.Render("{% set x = 1 -%}  hello", []);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Render_Comments_Stripped()
    {
        var result = JinjaTemplate.Render("before{# comment #}after", []);
        Assert.Equal("beforeafter", result);
    }

    [Fact]
    public void Render_FilterItems()
    {
        var dict = new Dictionary<string, object?> { ["a"] = 1, ["b"] = 2 };
        var result = JinjaTemplate.Render("{% for kv in d|items %}{{ kv[0] }}={{ kv[1] }} {% endfor %}", new Dictionary<string, object?> { ["d"] = dict });
        Assert.Equal("a=1 b=2 ", result);
    }

    [Fact]
    public void Render_NestedIf()
    {
        var result = JinjaTemplate.Render(
            "{% if a %}{% if b %}both{% else %}a-only{% endif %}{% else %}none{% endif %}",
            new Dictionary<string, object?> { ["a"] = true, ["b"] = true });
        Assert.Equal("both", result);
    }

    [Fact]
    public void Render_NoneCoalescing()
    {
        var result = JinjaTemplate.Render("{{ x|default('fallback') }}", new Dictionary<string, object?>
            { ["x"] = null });
        Assert.Equal("fallback", result);
    }

    [Fact]
    public void Render_ForLoopInMessages_Context()
    {
        var messages = new List<object?>
        {
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Hello" },
            new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "Hi there" },
        };

        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["bos_token"] = "<s>",
            ["eos_token"] = "</s>",
        };

        var result = JinjaTemplate.Render("{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}", ctx);
        Assert.Equal("user: Hello\nassistant: Hi there\n", result);
    }

    [Fact]
    public void Render_FilterToJson()
    {
        var result = JinjaTemplate.Render("{{ items|tojson }}", new Dictionary<string, object?>
            { ["items"] = new List<object?> { "a", "b" } });
        Assert.Equal("[\"a\", \"b\"]", result);
    }

    [Fact]
    public void Render_FilterLower()
    {
        var result = JinjaTemplate.Render("{{ 'HELLO'|lower }}", []);
        Assert.Equal("hello", result);
    }

    [Fact]
    public void Render_FilterUpper()
    {
        var result = JinjaTemplate.Render("{{ 'hello'|upper }}", []);
        Assert.Equal("HELLO", result);
    }

    [Fact]
    public void Render_ArithmeticOperators()
    {
        Assert.Equal("3", JinjaTemplate.Render("{{ 1 + 2 }}", []));
        Assert.Equal("-1", JinjaTemplate.Render("{{ 2 - 3 }}", []));
        Assert.Equal("6", JinjaTemplate.Render("{{ 2 * 3 }}", []));
        Assert.Equal("2", JinjaTemplate.Render("{{ 7 // 3 }}", []));
        Assert.Equal("1", JinjaTemplate.Render("{{ 7 % 3 }}", []));
    }

    [Fact]
    public void Render_BooleanNotOperator()
    {
        Assert.Equal("false", JinjaTemplate.Render("{{ not true }}", []));
        Assert.Equal("true", JinjaTemplate.Render("{{ not false }}", []));
    }

    [Fact]
    public void Render_Phi3Template_Diagnostic()
    {
        var messages = new List<object?>
        {
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Summarize this." },
        };

        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["bos_token"] = "<s>",
            ["eos_token"] = "<|end|>",
            ["add_generation_prompt"] = true,
        };

        var template = "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}<|end|>\n{% endfor %}<s>\n\n";
        var result = JinjaTemplate.Render(template, ctx);

        Assert.Contains("<|user|>", result);
    }

    [Fact]
    public void Render_Phi3Template_WithChatTemplateContext()
    {
        var tokens = new[] { "<pad>", "<s>", "<|end|>" };
        var scores = new float[] { 0f, 0f, 0f };
        var merges = Array.Empty<string>();
        var tokenizer = new BpeTokenizer(tokens, scores, merges, bosTokenId: 1, eosTokenId: 2);

        var template = "{% for message in messages %}<|{{ message.role }}|>\n{{ message.content }}<|end|>\n{% endfor %}<s>\n\n";
        var chatTemplate = ChatTemplate.FromGguf(
            new GgufMetadata(new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.chat_template"] = new GgufMetadataValue { Type = GgufValueType.String, Value = template },
            }),
            tokenizer)!;

        var result = chatTemplate.Format([
            new ChatMessageEntry("user", "Summarize this."),
        ]);

        var expected = "<|user|>\nSummarize this.<|end|>\n<s>\n\n";
        Assert.Equal(expected, result);
    }
}