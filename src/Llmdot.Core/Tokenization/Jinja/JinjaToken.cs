namespace Llmdot.Tokenization.Jinja;

internal enum JinjaTokenType
{
    Text,
    ExpressionStart,
    ExpressionEnd,
    StatementStart,
    StatementEnd,
    CommentStart,
    CommentEnd,
    Identifier,
    Integer,
    Float,
    String,
    Dot,
    Comma,
    Colon,
    Pipe,
    Assign,
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    DoubleStar,
    Tilde,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Not,
    And,
    Or,
    True,
    False,
    None,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    If,
    Elif,
    Else,
    EndIf,
    For,
    EndFor,
    In,
    Set,
    Is,
    Eof,
    NotIn,
}

internal sealed class JinjaToken
{
    public JinjaTokenType Type { get; init; }
    public string Value { get; init; } = string.Empty;

    public override string ToString() => $"{Type}: '{Value}'";
}