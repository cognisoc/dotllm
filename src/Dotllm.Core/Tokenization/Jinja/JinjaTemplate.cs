namespace Dotllm.Tokenization.Jinja;

internal static class JinjaTemplate
{
    public static string Render(string template, Dictionary<string, object?> context)
    {
        var tokens = new JinjaLexer(template).Tokenize();
        var ast = new JinjaParser(tokens).Parse();
        var evaluator = new JinjaEvaluator(context);
        return evaluator.Evaluate(ast);
    }
}