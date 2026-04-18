using System.Text;

namespace Dotllm.Cli;

internal static class CliOutput
{
    public static void EnsureUtf8()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;
    }

    public static void WriteError(string message)
    {
        Console.Error.WriteLine($"Error: {message}");
    }

    public static void WriteTable(IReadOnlyList<(string Key, string Value)> rows)
    {
        if (rows.Count == 0) return;
        var maxKey = rows.Max(r => r.Key.Length);
        foreach (var (key, value) in rows)
        {
            Console.Write(key.PadRight(maxKey + 2));
            Console.WriteLine(value);
        }
    }

    public static void WriteHeader(string text)
    {
        Console.WriteLine(text);
        Console.WriteLine(new string('-', text.Length));
    }
}
