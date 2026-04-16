using System.Text;

namespace Dotllm.Tokenization.Jinja;

internal sealed class JinjaLexer
{
    private readonly string _source;
    private int _pos;
    private bool _trimNextText;

    private enum BlockMode
    {
        Text,
        Expression,
        Statement,
        Comment,
    }

    public JinjaLexer(string source)
    {
        _source = source;
        _pos = 0;
    }

    public List<JinjaToken> Tokenize()
    {
        var tokens = new List<JinjaToken>();
        var mode = BlockMode.Text;

        while (_pos < _source.Length)
        {
            switch (mode)
            {
                case BlockMode.Text:
                    ReadTextBlock(tokens, ref mode);
                    break;
                case BlockMode.Expression:
                    ReadInnerBlock(tokens, "}}", JinjaTokenType.ExpressionEnd, ref mode);
                    break;
                case BlockMode.Statement:
                    ReadInnerBlock(tokens, "%}", JinjaTokenType.StatementEnd, ref mode);
                    break;
                case BlockMode.Comment:
                    SkipComment(ref mode);
                    break;
            }
        }

        tokens.Add(new JinjaToken { Type = JinjaTokenType.Eof });
        return tokens;
    }

    private void ReadTextBlock(List<JinjaToken> tokens, ref BlockMode mode)
    {
        var sb = new StringBuilder();

        if (_trimNextText)
        {
            _trimNextText = false;
            while (_pos < _source.Length && char.IsWhiteSpace(_source[_pos]) && !Matches("{{") && !Matches("{%") && !Matches("{#"))
                _pos++;
        }

        while (_pos < _source.Length)
        {
            if (Matches("{{-"))
            {
                TrimTrailingWhitespace(sb);
                _pos += 3;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                tokens.Add(new JinjaToken { Type = JinjaTokenType.ExpressionStart, Value = "{{-" });
                mode = BlockMode.Expression;
                return;
            }

            if (Matches("{%-"))
            {
                TrimTrailingWhitespace(sb);
                _pos += 3;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                tokens.Add(new JinjaToken { Type = JinjaTokenType.StatementStart, Value = "{%-" });
                mode = BlockMode.Statement;
                return;
            }

            if (Matches("{#-"))
            {
                TrimTrailingWhitespace(sb);
                _pos += 3;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                mode = BlockMode.Comment;
                return;
            }

            if (Matches("{{"))
            {
                _pos += 2;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                tokens.Add(new JinjaToken { Type = JinjaTokenType.ExpressionStart, Value = "{{" });
                mode = BlockMode.Expression;
                return;
            }

            if (Matches("{%"))
            {
                _pos += 2;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                tokens.Add(new JinjaToken { Type = JinjaTokenType.StatementStart, Value = "{%" });
                mode = BlockMode.Statement;
                return;
            }

            if (Matches("{#"))
            {
                _pos += 2;
                if (sb.Length > 0)
                    tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
                mode = BlockMode.Comment;
                return;
            }

            sb.Append(_source[_pos]);
            _pos++;
        }

        if (sb.Length > 0)
            tokens.Add(new JinjaToken { Type = JinjaTokenType.Text, Value = sb.ToString() });
    }

    private void ReadInnerBlock(List<JinjaToken> tokens, string endTag, JinjaTokenType endType, ref BlockMode mode)
    {
        SkipWhitespace();

        while (_pos < _source.Length)
        {
            SkipWhitespace();

            if (Matches("-" + endTag))
            {
                _pos += endTag.Length + 1;
                tokens.Add(new JinjaToken { Type = endType, Value = "-" + endTag });
                mode = BlockMode.Text;
                _trimNextText = true;
                return;
            }

            if (Matches(endTag))
            {
                _pos += endTag.Length;
                tokens.Add(new JinjaToken { Type = endType, Value = endTag });
                mode = BlockMode.Text;
                return;
            }

            var token = ReadToken();
            if (token is not null)
                tokens.Add(token);
        }
    }

    private void SkipComment(ref BlockMode mode)
    {
        while (_pos < _source.Length)
        {
            if (Matches("-#}"))
            {
                _pos += 3;
                _trimNextText = true;
                mode = BlockMode.Text;
                return;
            }

            if (Matches("#}"))
            {
                _pos += 2;
                mode = BlockMode.Text;
                return;
            }

            _pos++;
        }
    }

    private JinjaToken? ReadToken()
    {
        if (_pos >= _source.Length) return null;

        var ch = _source[_pos];

        if (ch == '\'' || ch == '"')
            return ReadString();

        if (char.IsDigit(ch))
            return ReadNumber();

        if (char.IsLetter(ch) || ch == '_')
            return ReadIdentifierOrKeyword();

        return ReadSymbol();
    }

    private JinjaToken ReadString()
    {
        var quote = _source[_pos];
        _pos++;
        var sb = new StringBuilder();

        while (_pos < _source.Length)
        {
            if (_source[_pos] == '\\' && _pos + 1 < _source.Length)
            {
                var next = _source[_pos + 1];
                _pos += 2;
                sb.Append(next switch
                {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => $"\\{next}",
                });
                continue;
            }

            if (_source[_pos] == quote)
            {
                _pos++;
                return new JinjaToken { Type = JinjaTokenType.String, Value = sb.ToString() };
            }

            sb.Append(_source[_pos]);
            _pos++;
        }

        return new JinjaToken { Type = JinjaTokenType.String, Value = sb.ToString() };
    }

    private JinjaToken ReadNumber()
    {
        var start = _pos;
        var isFloat = false;

        while (_pos < _source.Length && (char.IsDigit(_source[_pos]) || _source[_pos] == '.'))
        {
            if (_source[_pos] == '.')
            {
                if (isFloat) break;
                isFloat = true;
            }
            _pos++;
        }

        var text = _source[start.._pos];
        return new JinjaToken { Type = isFloat ? JinjaTokenType.Float : JinjaTokenType.Integer, Value = text };
    }

    private JinjaToken ReadIdentifierOrKeyword()
    {
        var start = _pos;
        while (_pos < _source.Length && (char.IsLetterOrDigit(_source[_pos]) || _source[_pos] == '_'))
            _pos++;

        var text = _source[start.._pos];
        var type = text switch
        {
            "if" => JinjaTokenType.If,
            "elif" => JinjaTokenType.Elif,
            "else" => JinjaTokenType.Else,
            "endif" => JinjaTokenType.EndIf,
            "for" => JinjaTokenType.For,
            "endfor" => JinjaTokenType.EndFor,
            "in" => JinjaTokenType.In,
            "set" => JinjaTokenType.Set,
            "is" => JinjaTokenType.Is,
            "not" => JinjaTokenType.Not,
            "and" => JinjaTokenType.And,
            "or" => JinjaTokenType.Or,
            "true" or "True" => JinjaTokenType.True,
            "false" or "False" => JinjaTokenType.False,
            "none" or "None" => JinjaTokenType.None,
            _ => JinjaTokenType.Identifier,
        };

        return new JinjaToken { Type = type, Value = text };
    }

    private JinjaToken? ReadSymbol()
    {
        var ch = _source[_pos];
        _pos++;

        var (type, value) = ch switch
        {
            '.' => (JinjaTokenType.Dot, "."),
            ',' => (JinjaTokenType.Comma, ","),
            ':' => (JinjaTokenType.Colon, ":"),
            '|' => (JinjaTokenType.Pipe, "|"),
            '=' => PeekAndConsume('=', out var eq) ? (JinjaTokenType.Eq, "==") : (JinjaTokenType.Assign, "="),
            '!' => PeekAndConsume('=', out _) ? (JinjaTokenType.Ne, "!=") : throw new InvalidOperationException($"Unexpected character '!' at position {_pos}"),
            '<' => PeekAndConsume('=', out _) ? (JinjaTokenType.Le, "<=") : (JinjaTokenType.Lt, "<"),
            '>' => PeekAndConsume('=', out _) ? (JinjaTokenType.Ge, ">=") : (JinjaTokenType.Gt, ">"),
            '+' => (JinjaTokenType.Plus, "+"),
            '-' => (JinjaTokenType.Minus, "-"),
            '*' => PeekAndConsume('*', out _) ? (JinjaTokenType.DoubleStar, "**") : (JinjaTokenType.Star, "*"),
            '/' => PeekAndConsume('/', out _) ? (JinjaTokenType.DoubleSlash, "//") : (JinjaTokenType.Slash, "/"),
            '%' => (JinjaTokenType.Percent, "%"),
            '~' => (JinjaTokenType.Tilde, "~"),
            '(' => (JinjaTokenType.LParen, "("),
            ')' => (JinjaTokenType.RParen, ")"),
            '[' => (JinjaTokenType.LBracket, "["),
            ']' => (JinjaTokenType.RBracket, "]"),
            '{' => (JinjaTokenType.LBrace, "{"),
            '}' => (JinjaTokenType.RBrace, "}"),
            _ => throw new InvalidOperationException($"Unexpected character '{ch}' at position {_pos}"),
        };

        return new JinjaToken { Type = type, Value = value };
    }

    private bool PeekAndConsume(char expected, out char consumed)
    {
        if (_pos < _source.Length && _source[_pos] == expected)
        {
            consumed = _source[_pos];
            _pos++;
            return true;
        }

        consumed = '\0';
        return false;
    }

    private bool Matches(string s)
    {
        if (_pos + s.Length > _source.Length) return false;
        for (var i = 0; i < s.Length; i++)
            if (_source[_pos + i] != s[i]) return false;
        return true;
    }

    private void SkipWhitespace()
    {
        while (_pos < _source.Length && char.IsWhiteSpace(_source[_pos]))
            _pos++;
    }

    private static void TrimTrailingWhitespace(StringBuilder sb)
    {
        while (sb.Length > 0 && char.IsWhiteSpace(sb[sb.Length - 1]))
            sb.Length--;
    }
}