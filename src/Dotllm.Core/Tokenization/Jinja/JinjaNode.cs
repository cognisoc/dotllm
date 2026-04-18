namespace Dotllm.Tokenization.Jinja;

internal abstract class JinjaNode;

internal sealed class TextNode : JinjaNode
{
    public string Text { get; }
    public TextNode(string text) => Text = text;
}

internal sealed class ExpressionNode : JinjaNode
{
    public JinjaExpr Expr { get; }
    public ExpressionNode(JinjaExpr expr) => Expr = expr;
}

internal sealed class IfNode : JinjaNode
{
    public JinjaExpr Condition { get; }
    public List<JinjaNode> Body { get; }
    public List<(JinjaExpr Condition, List<JinjaNode> Body)> ElifBranches { get; }
    public List<JinjaNode> ElseBody { get; }

    public IfNode(JinjaExpr condition, List<JinjaNode> body, List<(JinjaExpr Condition, List<JinjaNode> Body)> elifBranches, List<JinjaNode> elseBody)
    {
        Condition = condition;
        Body = body;
        ElifBranches = elifBranches;
        ElseBody = elseBody;
    }
}

internal sealed class ForNode : JinjaNode
{
    public string LoopVar { get; }
    public JinjaExpr Iterable { get; }
    public List<JinjaNode> Body { get; }
    public bool IsRecursive { get; }

    public ForNode(string loopVar, JinjaExpr iterable, List<JinjaNode> body, bool isRecursive = false)
    {
        LoopVar = loopVar;
        Iterable = iterable;
        Body = body;
        IsRecursive = isRecursive;
    }
}

internal sealed class SetNode : JinjaNode
{
    public string Name { get; }
    public string? Attr { get; }
    public JinjaExpr Value { get; }

    public SetNode(string name, JinjaExpr value)
    {
        Name = name;
        Value = value;
    }

    public SetNode(string name, string attr, JinjaExpr value)
    {
        Name = name;
        Attr = attr;
        Value = value;
    }
}

internal abstract class JinjaExpr;

internal sealed class LiteralExpr : JinjaExpr
{
    public object? Value { get; }
    public LiteralExpr(object? value) => Value = value;
}

internal sealed class NameExpr : JinjaExpr
{
    public string Name { get; }
    public NameExpr(string name) => Name = name;
}

internal sealed class GetAttrExpr : JinjaExpr
{
    public JinjaExpr Object { get; }
    public string Attr { get; }
    public GetAttrExpr(JinjaExpr obj, string attr) { Object = obj; Attr = attr; }
}

internal sealed class GetItemExpr : JinjaExpr
{
    public JinjaExpr Object { get; }
    public JinjaExpr Key { get; }
    public GetItemExpr(JinjaExpr obj, JinjaExpr key) { Object = obj; Key = key; }
}

internal sealed class BinOpExpr : JinjaExpr
{
    public string Op { get; }
    public JinjaExpr Left { get; }
    public JinjaExpr Right { get; }
    public BinOpExpr(string op, JinjaExpr left, JinjaExpr right) { Op = op; Left = left; Right = right; }
}

internal sealed class UnaryOpExpr : JinjaExpr
{
    public string Op { get; }
    public JinjaExpr Operand { get; }
    public UnaryOpExpr(string op, JinjaExpr operand) { Op = op; Operand = operand; }
}

internal sealed class FilterExpr : JinjaExpr
{
    public JinjaExpr Value { get; }
    public string Name { get; }
    public List<JinjaExpr> Args { get; }
    public FilterExpr(JinjaExpr value, string name, List<JinjaExpr> args) { Value = value; Name = name; Args = args; }
}

internal sealed class TestExpr : JinjaExpr
{
    public JinjaExpr Value { get; }
    public string Name { get; }
    public bool Negated { get; }
    public TestExpr(JinjaExpr value, string name, bool negated = false) { Value = value; Name = name; Negated = negated; }
}

internal sealed class CondExpr : JinjaExpr
{
    public JinjaExpr Condition { get; }
    public JinjaExpr TrueExpr { get; }
    public JinjaExpr FalseExpr { get; }
    public CondExpr(JinjaExpr condition, JinjaExpr trueExpr, JinjaExpr falseExpr) { Condition = condition; TrueExpr = trueExpr; FalseExpr = falseExpr; }
}

internal sealed class CallExpr : JinjaExpr
{
    public JinjaExpr Function { get; }
    public List<JinjaExpr> Args { get; }
    public Dictionary<string, JinjaExpr> Kwargs { get; }
    public CallExpr(JinjaExpr function, List<JinjaExpr> args, Dictionary<string, JinjaExpr> kwargs) { Function = function; Args = args; Kwargs = kwargs; }
}

internal sealed class ConcatExpr : JinjaExpr
{
    public JinjaExpr Left { get; }
    public JinjaExpr Right { get; }
    public ConcatExpr(JinjaExpr left, JinjaExpr right) { Left = left; Right = right; }
}

internal sealed class SliceExpr : JinjaExpr
{
    public JinjaExpr Object { get; }
    public JinjaExpr? Start { get; }
    public JinjaExpr? End { get; }

    public SliceExpr(JinjaExpr obj, JinjaExpr? start, JinjaExpr? end)
    {
        Object = obj;
        Start = start;
        End = end;
    }
}