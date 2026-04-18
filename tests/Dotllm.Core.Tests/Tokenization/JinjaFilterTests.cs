using Dotllm.Tokenization.Jinja;
using Xunit;

namespace Dotllm.Core.Tests.Tokenization;

public class JinjaFilterTests
{
    [Fact]
    public void Map_Attribute_ExtractsField()
    {
        var messages = new List<object?>
        {
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Hello" },
            new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "Hi" },
        };
        var result = JinjaTemplate.Render("{{ messages|map('role')|join(', ') }}", new Dictionary<string, object?> { ["messages"] = messages });
        Assert.Equal("user, assistant", result);
    }

    [Fact]
    public void Map_Filter_AppliesFilterToEach()
    {
        var names = new List<object?> { "alice", "bob" };
        var result = JinjaTemplate.Render("{{ names|map('upper')|join(', ') }}", new Dictionary<string, object?> { ["names"] = names });
        Assert.Equal("ALICE, BOB", result);
    }

    [Fact]
    public void SelectAttr_FiltersByAttribute()
    {
        var messages = new List<object?>
        {
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Hello" },
            new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "Hi" },
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Bye" },
        };
        var result = JinjaTemplate.Render("{{ messages|selectattr('role', 'equalto', 'user')|length }}", new Dictionary<string, object?> { ["messages"] = messages });
        Assert.Equal("2", result);
    }

    [Fact]
    public void RejectAttr_FiltersByAttribute()
    {
        var messages = new List<object?>
        {
            new Dictionary<string, object?> { ["role"] = "system", ["content"] = "You are a bot" },
            new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Hello" },
            new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = "Hi" },
        };
        var result = JinjaTemplate.Render("{{ messages|rejectattr('role', 'equalto', 'system')|length }}", new Dictionary<string, object?> { ["messages"] = messages });
        Assert.Equal("2", result);
    }

    [Fact]
    public void Reject_SupportsNoneTest()
    {
        var items = new List<object?> { "a", null, "b", null };
        var result = JinjaTemplate.Render("{{ items|reject('none')|join(', ') }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("a, b", result);
    }

    [Fact]
    public void Sort_DefaultAscending()
    {
        var items = new List<object?> { 3, 1, 2 };
        var result = JinjaTemplate.Render("{{ items|sort|join(', ') }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("1, 2, 3", result);
    }

    [Fact]
    public void Sort_Reverse()
    {
        var items = new List<object?> { 1, 2, 3 };
        var result = JinjaTemplate.Render("{{ items|sort(true)|join(', ') }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("3, 2, 1", result);
    }

    [Fact]
    public void Reverse_List()
    {
        var items = new List<object?> { 1, 2, 3 };
        var result = JinjaTemplate.Render("{{ items|reverse|join(', ') }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("3, 2, 1", result);
    }

    [Fact]
    public void Unique_RemovesDuplicates()
    {
        var items = new List<object?> { 1, 2, 2, 3 };
        var result = JinjaTemplate.Render("{{ items|unique|join(', ') }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("1, 2, 3", result);
    }

    [Fact]
    public void Int_CastsString()
    {
        var result = JinjaTemplate.Render("{{ '42'|int }}", new Dictionary<string, object?>());
        Assert.Equal("42", result);
    }

    [Fact]
    public void Float_CastsString()
    {
        var result = JinjaTemplate.Render("{{ '3.14'|float }}", new Dictionary<string, object?>());
        Assert.Contains("3.14", result);
    }

    [Fact]
    public void Abs_NegativeToPositive()
    {
        var result = JinjaTemplate.Render("{{ x|abs }}", new Dictionary<string, object?> { ["x"] = -5 });
        Assert.Equal("5", result);
    }

    [Fact]
    public void Round_DefaultZeroPrecision()
    {
        var result = JinjaTemplate.Render("{{ x|round }}", new Dictionary<string, object?> { ["x"] = 3.7 });
        Assert.Equal("4", result);
    }

    [Fact]
    public void Indent_AddsWhitespace()
    {
        var result = JinjaTemplate.Render("{{ text|indent(2) }}", new Dictionary<string, object?> { ["text"] = "line1\nline2\nline3" });
        Assert.Equal("line1\n  line2\n  line3", result);
    }

    [Fact]
    public void Title_TitleCases()
    {
        var result = JinjaTemplate.Render("{{ 'hello world'|title }}", new Dictionary<string, object?>());
        Assert.Equal("Hello World", result);
    }

    [Fact]
    public void Capitalize_FirstLetter()
    {
        var result = JinjaTemplate.Render("{{ 'hello'|capitalize }}", new Dictionary<string, object?>());
        Assert.Equal("Hello", result);
    }

    [Fact]
    public void RaiseException_ThrowsJinjaException()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["raise_exception"] = (Func<object?[], Dictionary<string, object?>, object?>)((args, _) =>
                throw new InvalidOperationException(args.Length > 0 ? args[0]?.ToString() ?? "Template error" : "Template error")),
        };
        Assert.Throws<InvalidOperationException>(() =>
            JinjaTemplate.Render("{{ raise_exception('test error') }}", ctx));
    }

    [Fact]
    public void Count_IsLengthAlias()
    {
        var items = new List<object?> { "a", "b", "c" };
        var result = JinjaTemplate.Render("{{ items|count }}", new Dictionary<string, object?> { ["items"] = items });
        Assert.Equal("3", result);
    }
}
