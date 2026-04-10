using System.Globalization;

namespace Dotllm.Tokenization.Jinja;

internal sealed class JinjaParser
{
    private readonly List<JinjaToken> _tokens;
    private int _pos;

    public JinjaParser(List<JinjaToken> tokens)
    {
        _tokens = tokens;
        _pos = 0;
    }

    public List<JinjaNode> Parse()
    {
        var nodes = new List<JinjaNode>();
        while (Current().Type != JinjaTokenType.Eof)
            nodes.Add(ParseNode());
        return nodes;
    }

    private JinjaNode ParseNode()
    {
        var token = Current();

        if (token.Type == JinjaTokenType.Text)
        {
            Advance();
            return new TextNode(token.Value);
        }

        if (token.Type == JinjaTokenType.ExpressionStart)
        {
            Advance();
            var expr = ParseExpression();
            Expect(JinjaTokenType.ExpressionEnd);
            return new ExpressionNode(expr);
        }

        if (token.Type == JinjaTokenType.StatementStart)
        {
            Advance();
            return ParseStatement();
        }

        throw new InvalidOperationException($"Unexpected token: {token}");
    }

    private JinjaNode ParseStatement()
    {
        var token = Current();

        if (token.Type == JinjaTokenType.If)
        {
            Advance();
            return ParseIf();
        }

        if (token.Type == JinjaTokenType.For)
        {
            Advance();
            return ParseFor();
        }

        if (token.Type == JinjaTokenType.Set)
        {
            Advance();
            return ParseSet();
        }

        throw new InvalidOperationException($"Unexpected statement keyword: {token}");
    }

    private IfNode ParseIf()
    {
        var condition = ParseExpression();
        Expect(JinjaTokenType.StatementEnd);

        var body = ParseBody(JinjaTokenType.Elif, JinjaTokenType.Else, JinjaTokenType.EndIf);
        var elifBranches = new List<(JinjaExpr Condition, List<JinjaNode> Body)>();
        var elseBody = new List<JinjaNode>();

        while (Current().Type == JinjaTokenType.StatementStart && PeekStatementType() == JinjaTokenType.Elif)
        {
            Advance();
            Advance();
            var elifCond = ParseExpression();
            Expect(JinjaTokenType.StatementEnd);
            var elifBody = ParseBody(JinjaTokenType.Elif, JinjaTokenType.Else, JinjaTokenType.EndIf);
            elifBranches.Add((elifCond, elifBody));
        }

        if (Current().Type == JinjaTokenType.StatementStart && PeekStatementType() == JinjaTokenType.Else)
        {
            Advance();
            Advance();
            Expect(JinjaTokenType.StatementEnd);
            elseBody = ParseBody(JinjaTokenType.EndIf);
        }

        ExpectStatement(JinjaTokenType.EndIf);
        return new IfNode(condition, body, elifBranches, elseBody);
    }

    private ForNode ParseFor()
    {
        var loopVar = Expect(JinjaTokenType.Identifier).Value;
        Expect(JinjaTokenType.In);
        var iterable = ParseExpression();
        Expect(JinjaTokenType.StatementEnd);

        var body = ParseBody(JinjaTokenType.EndFor);
        ExpectStatement(JinjaTokenType.EndFor);
        return new ForNode(loopVar, iterable, body);
    }

    private SetNode ParseSet()
    {
        var name = Expect(JinjaTokenType.Identifier).Value;
        Expect(JinjaTokenType.Assign);
        var value = ParseExpression();
        Expect(JinjaTokenType.StatementEnd);
        return new SetNode(name, value);
    }

    private List<JinjaNode> ParseBody(params JinjaTokenType[] stopTokens)
    {
        var nodes = new List<JinjaNode>();

        while (true)
        {
            var token = Current();
            if (token.Type == JinjaTokenType.Eof)
                break;

            if (token.Type == JinjaTokenType.StatementStart)
            {
                var nextType = PeekStatementType();
                if (Array.IndexOf(stopTokens, nextType) >= 0)
                    break;
            }

            nodes.Add(ParseNode());
        }

        return nodes;
    }

    private JinjaTokenType PeekStatementType()
    {
        var saved = _pos;
        Advance();
        var type = Current().Type;
        _pos = saved;
        return type;
    }

    private void ExpectStatement(JinjaTokenType keywordType)
    {
        Expect(JinjaTokenType.StatementStart);
        Expect(keywordType);
        Expect(JinjaTokenType.StatementEnd);
    }

    private JinjaExpr ParseExpression()
    {
        return ParseTernary();
    }

    private JinjaExpr ParseTernary()
    {
        var expr = ParseOr();

        if (Current().Type == JinjaTokenType.If || (Current().Type == JinjaTokenType.Identifier && Current().Value == "if"))
        {
            Advance();
            var condition = ParseOr();

            if (Current().Type == JinjaTokenType.Else || (Current().Type == JinjaTokenType.Identifier && Current().Value == "else"))
            {
                Advance();
                var falseExpr = ParseTernary();
                return new CondExpr(condition, expr, falseExpr);
            }

            return new CondExpr(condition, expr, new LiteralExpr(""));
        }

        return expr;
    }

    private JinjaExpr ParseOr()
    {
        var left = ParseAnd();
        while (Current().Type == JinjaTokenType.Or)
        {
            Advance();
            var right = ParseAnd();
            left = new BinOpExpr("or", left, right);
        }
        return left;
    }

    private JinjaExpr ParseAnd()
    {
        var left = ParseNot();
        while (Current().Type == JinjaTokenType.And)
        {
            Advance();
            var right = ParseNot();
            left = new BinOpExpr("and", left, right);
        }
        return left;
    }

    private JinjaExpr ParseNot()
    {
        if (Current().Type == JinjaTokenType.Not)
        {
            Advance();
            return new UnaryOpExpr("not", ParseNot());
        }
        return ParseComparison();
    }

    private JinjaExpr ParseComparison()
    {
        var left = ParseConcat();

        while (Current().Type is JinjaTokenType.Eq or JinjaTokenType.Ne or JinjaTokenType.Lt
            or JinjaTokenType.Gt or JinjaTokenType.Le or JinjaTokenType.Ge)
        {
            var op = Current().Value;
            Advance();
            var right = ParseConcat();
            left = new BinOpExpr(op, left, right);
        }

        if (Current().Type == JinjaTokenType.Is)
        {
            Advance();
            var negated = false;
            if (Current().Type == JinjaTokenType.Not)
            {
                negated = true;
                Advance();
            }
            var testName = Current().Type switch
            {
                JinjaTokenType.Identifier => Current().Value,
                JinjaTokenType.None => "none",
                JinjaTokenType.True => "true",
                JinjaTokenType.False => "false",
                _ => Current().Value,
            };
            Advance();
            left = new TestExpr(left, testName, negated);
        }

        return left;
    }

    private JinjaExpr ParseConcat()
    {
        var left = ParseAdditive();
        while (Current().Type == JinjaTokenType.Tilde)
        {
            Advance();
            var right = ParseAdditive();
            left = new ConcatExpr(left, right);
        }
        return left;
    }

    private JinjaExpr ParseAdditive()
    {
        var left = ParseMultiplicative();
        while (Current().Type is JinjaTokenType.Plus or JinjaTokenType.Minus)
        {
            var op = Current().Value;
            Advance();
            var right = ParseMultiplicative();
            left = new BinOpExpr(op, left, right);
        }
        return left;
    }

    private JinjaExpr ParseMultiplicative()
    {
        var left = ParseUnary();
        while (Current().Type is JinjaTokenType.Star or JinjaTokenType.Slash
            or JinjaTokenType.DoubleSlash or JinjaTokenType.Percent)
        {
            var op = Current().Value;
            Advance();
            var right = ParseUnary();
            left = new BinOpExpr(op, left, right);
        }
        return left;
    }

    private JinjaExpr ParseUnary()
    {
        if (Current().Type == JinjaTokenType.Minus)
        {
            Advance();
            return new UnaryOpExpr("-", ParseUnary());
        }
        return ParsePower();
    }

    private JinjaExpr ParsePower()
    {
        var left = ParsePostfix();
        if (Current().Type == JinjaTokenType.DoubleStar)
        {
            Advance();
            var right = ParseUnary();
            return new BinOpExpr("**", left, right);
        }
        return left;
    }

    private JinjaExpr ParsePostfix()
    {
        var expr = ParsePrimary();

        while (true)
        {
            if (Current().Type == JinjaTokenType.Dot)
            {
                Advance();
                var attr = Expect(JinjaTokenType.Identifier).Value;
                expr = new GetAttrExpr(expr, attr);
            }
            else if (Current().Type == JinjaTokenType.LBracket)
            {
                Advance();
                var key = ParseExpression();
                Expect(JinjaTokenType.RBracket);
                expr = new GetItemExpr(expr, key);
            }
            else if (Current().Type == JinjaTokenType.LParen)
            {
                Advance();
                var (args, kwargs) = ParseCallArgs();
                Expect(JinjaTokenType.RParen);
                expr = new CallExpr(expr, args, kwargs);
            }
            else if (Current().Type == JinjaTokenType.Pipe)
            {
                Advance();
                var filterName = Expect(JinjaTokenType.Identifier).Value;
                List<JinjaExpr> filterArgs = [];
                if (Current().Type == JinjaTokenType.LParen)
                {
                    Advance();
                    var (args, kwargs) = ParseCallArgs();
                    Expect(JinjaTokenType.RParen);
                    filterArgs = args;
                }
                expr = new FilterExpr(expr, filterName, filterArgs);
            }
            else
            {
                break;
            }
        }

        return expr;
    }

    private JinjaExpr ParsePrimary()
    {
        var token = Current();

        switch (token.Type)
        {
            case JinjaTokenType.Integer:
                Advance();
                return new LiteralExpr(int.Parse(token.Value, CultureInfo.InvariantCulture));

            case JinjaTokenType.Float:
                Advance();
                return new LiteralExpr(float.Parse(token.Value, System.Globalization.CultureInfo.InvariantCulture));

            case JinjaTokenType.String:
                Advance();
                return new LiteralExpr(token.Value);

            case JinjaTokenType.True:
                Advance();
                return new LiteralExpr(true);

            case JinjaTokenType.False:
                Advance();
                return new LiteralExpr(false);

            case JinjaTokenType.None:
                Advance();
                return new LiteralExpr(null);

            case JinjaTokenType.Identifier:
                Advance();
                return new NameExpr(token.Value);

            case JinjaTokenType.LParen:
                Advance();
                var expr = ParseExpression();
                Expect(JinjaTokenType.RParen);
                return expr;

            case JinjaTokenType.LBracket:
                Advance();
                return ParseListLiteral();

            case JinjaTokenType.LBrace:
                Advance();
                return ParseDictLiteral();

            default:
                throw new InvalidOperationException($"Unexpected token in expression: {token}");
        }
    }

    private LiteralExpr ParseListLiteral()
    {
        var items = new List<JinjaExpr>();
        while (Current().Type != JinjaTokenType.RBracket)
        {
            items.Add(ParseExpression());
            if (Current().Type == JinjaTokenType.Comma) Advance();
        }
        Expect(JinjaTokenType.RBracket);
        return new LiteralExpr(items);
    }

    private LiteralExpr ParseDictLiteral()
    {
        var keys = new List<JinjaExpr>();
        var values = new List<JinjaExpr>();
        while (Current().Type != JinjaTokenType.RBrace)
        {
            keys.Add(ParseExpression());
            Expect(JinjaTokenType.Colon);
            values.Add(ParseExpression());
            if (Current().Type == JinjaTokenType.Comma) Advance();
        }
        Expect(JinjaTokenType.RBrace);
        return new LiteralExpr((keys, values));
    }

    private (List<JinjaExpr> args, Dictionary<string, JinjaExpr> kwargs) ParseCallArgs()
    {
        var args = new List<JinjaExpr>();
        var kwargs = new Dictionary<string, JinjaExpr>();

        while (Current().Type != JinjaTokenType.RParen)
        {
            if (Current().Type == JinjaTokenType.Identifier &&
                _pos + 1 < _tokens.Count &&
                _tokens[_pos + 1].Type == JinjaTokenType.Assign)
            {
                var name = Current().Value;
                Advance();
                Advance();
                kwargs[name] = ParseExpression();
            }
            else
            {
                args.Add(ParseExpression());
            }

            if (Current().Type == JinjaTokenType.Comma) Advance();
        }

        return (args, kwargs);
    }

    private JinjaToken Current() => _pos < _tokens.Count ? _tokens[_pos] : new JinjaToken { Type = JinjaTokenType.Eof };

    private JinjaToken Advance()
    {
        var token = Current();
        if (_pos < _tokens.Count) _pos++;
        return token;
    }

    private JinjaToken Expect(JinjaTokenType type)
    {
        var token = Current();
        if (token.Type != type)
            throw new InvalidOperationException($"Expected {type}, got {token}");
        Advance();
        return token;
    }
}