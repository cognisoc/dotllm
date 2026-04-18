using Llmdot.Cli;
using Llmdot.Cli.Commands;

CliOutput.EnsureUtf8();

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cts.Cancel();
};

if (args.Length == 0)
{
    PrintUsage();
    return 1;
}

var command = args[0];

return command switch
{
    "info" when args.Length >= 2 => InfoCommand.Run(args[1]),
    "chat" when args.Length >= 2 => await ChatCommand.RunAsync(
        args[1],
        args.Length >= 3 && !args[2].StartsWith("--") ? args[2] : null,
        CommandOptions.Parse(args[2..]),
        cts.Token),
    "complete" when args.Length >= 3 => await CompleteCommand.RunAsync(
        args[1], args[2], CommandOptions.Parse(args[3..]), cts.Token),
    "--help" or "-h" => PrintUsageReturn(),
    _ => PrintUnknownCommand(command),
};

static void PrintUsage()
{
    Console.WriteLine("Usage: llmdot <command> [options]");
    Console.WriteLine();
    Console.WriteLine("Commands:");
    Console.WriteLine("  info <model.gguf>                          Show model metadata");
    Console.WriteLine("  chat <model.gguf> [prompt] [options]       Interactive or one-shot chat");
    Console.WriteLine("  complete <model.gguf> <prompt> [options]   Raw text completion");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --max-tokens N        Maximum tokens to generate (default: 256)");
    Console.WriteLine("  --temperature F       Sampling temperature (default: 0.8, 0=greedy)");
    Console.WriteLine("  --top-k N             Top-K sampling (default: 40)");
    Console.WriteLine("  --top-p F             Top-P/nucleus sampling (default: 0.95)");
    Console.WriteLine("  --repeat-penalty F    Repeat penalty (default: 1.1)");
    Console.WriteLine("  --seed N              Random seed (-1=random, default: -1)");
}

static int PrintUsageReturn() { PrintUsage(); return 0; }

static int PrintUnknownCommand(string cmd)
{
    CliOutput.WriteError($"Unknown command: '{cmd}'. Run 'llmdot --help' for usage.");
    return 1;
}
