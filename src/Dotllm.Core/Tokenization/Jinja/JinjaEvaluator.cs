using System.Globalization;
using System.Text;

namespace Dotllm.Tokenization.Jinja;

internal sealed class JinjaEvaluator
{
    private readonly Dictionary<string, object?> _globals;
    private readonly Dictionary<string, Func<object?[], object?>> _filters;
    private readonly Dictionary<string, Func<object?, Dictionary<string, object?>, bool>> _tests;

    public JinjaEvaluator(Dictionary<string, object?> globals)
    {
        _globals = globals;
        _filters = CreateFilters();
        _tests = CreateTests();
    }

    public string Evaluate(List<JinjaNode> nodes)
    {
        var sb = new StringBuilder();
        var scope = new Scope(_globals);

        foreach (var node in nodes)
            sb.Append(EvalNode(node, scope));

        return sb.ToString();
    }

    private string EvalNode(JinjaNode node, Scope scope)
    {
        switch (node)
        {
            case TextNode text:
                return text.Text;
            case ExpressionNode expr:
                return Stringify(EvalExpr(expr.Expr, scope));
            case IfNode ifNode:
                return EvalIf(ifNode, scope);
            case ForNode forNode:
                return EvalFor(forNode, scope);
            case SetNode setNode:
                scope[setNode.Name] = EvalExpr(setNode.Value, scope);
                return "";
            default:
                throw new InvalidOperationException($"Unknown node type: {node.GetType().Name}");
        }
    }

    private string EvalIf(IfNode node, Scope scope)
    {
        if (IsTruthy(EvalExpr(node.Condition, scope)))
            return EvalBody(node.Body, scope);

        foreach (var (condition, body) in node.ElifBranches)
        {
            if (IsTruthy(EvalExpr(condition, scope)))
                return EvalBody(body, scope);
        }

        return EvalBody(node.ElseBody, scope);
    }

    private string EvalFor(ForNode node, Scope scope)
    {
        var iterable = EvalExpr(node.Iterable, scope);
        var items = ToList(iterable);
        var sb = new StringBuilder();

        for (var i = 0; i < items.Count; i++)
        {
            var childScope = scope.Push();
            childScope[node.LoopVar] = items[i];
            childScope["loop"] = new Dictionary<string, object?>
            {
                ["index"] = i + 1,
                ["index0"] = i,
                ["first"] = i == 0,
                ["last"] = i == items.Count - 1,
                ["length"] = items.Count,
            };
            sb.Append(EvalBody(node.Body, childScope));
        }

        return sb.ToString();
    }

    private string EvalBody(List<JinjaNode> body, Scope scope)
    {
        var sb = new StringBuilder();
        foreach (var node in body)
            sb.Append(EvalNode(node, scope));
        return sb.ToString();
    }

    private object? EvalExpr(JinjaExpr expr, Scope scope)
    {
        switch (expr)
        {
            case LiteralExpr lit:
                return lit.Value;
            case NameExpr name:
                return scope.TryGetValue(name.Name, out var val) ? val : null;
            case GetAttrExpr attr:
                {
                    var obj = EvalExpr(attr.Object, scope);
                    return GetAttr(obj, attr.Attr);
                }
            case GetItemExpr item:
                {
                    var obj = EvalExpr(item.Object, scope);
                    var key = EvalExpr(item.Key, scope);
                    return GetItem(obj, key);
                }
            case BinOpExpr bin:
                return EvalBinOp(bin, scope);
            case UnaryOpExpr unary:
                return EvalUnaryOp(unary, scope);
            case FilterExpr filter:
                return EvalFilter(filter, scope);
            case TestExpr test:
                return EvalTest(test, scope);
            case CondExpr cond:
                return IsTruthy(EvalExpr(cond.Condition, scope))
                    ? EvalExpr(cond.TrueExpr, scope)
                    : EvalExpr(cond.FalseExpr, scope);
            case CallExpr call:
                return EvalCall(call, scope);
            case ConcatExpr concat:
                return Stringify(EvalExpr(concat.Left, scope)) + Stringify(EvalExpr(concat.Right, scope));
            default:
                throw new InvalidOperationException($"Unknown expression type: {expr.GetType().Name}");
        }
    }

    private object? EvalBinOp(BinOpExpr bin, Scope scope)
    {
        if (bin.Op == "and") return IsTruthy(EvalExpr(bin.Left, scope)) ? EvalExpr(bin.Right, scope) : EvalExpr(bin.Left, scope);
        if (bin.Op == "or") return IsTruthy(EvalExpr(bin.Left, scope)) ? EvalExpr(bin.Left, scope) : EvalExpr(bin.Right, scope);

        var left = EvalExpr(bin.Left, scope);
        var right = EvalExpr(bin.Right, scope);

        return bin.Op switch
        {
            "+" when left is string ls && right is string rs => ls + rs,
            "+" => ToDouble(left) + ToDouble(right),
            "-" => ToDouble(left) - ToDouble(right),
            "*" => ToDouble(left) * ToDouble(right),
            "/" => ToDouble(right) != 0 ? ToDouble(left) / ToDouble(right) : 0,
            "//" => ToDouble(right) != 0 ? Math.Floor(ToDouble(left) / ToDouble(right)) : 0,
            "%" => ToDouble(right) != 0 ? ToDouble(left) % ToDouble(right) : 0,
            "**" => Math.Pow(ToDouble(left), ToDouble(right)),
            "==" => Equals(left, right),
            "!=" => !Equals(left, right),
            "<" => Compare(left, right) < 0,
            ">" => Compare(left, right) > 0,
            "<=" => Compare(left, right) <= 0,
            ">=" => Compare(left, right) >= 0,
            _ => throw new InvalidOperationException($"Unknown operator: {bin.Op}"),
        };
    }

    private object? EvalUnaryOp(UnaryOpExpr unary, Scope scope)
    {
        var val = EvalExpr(unary.Operand, scope);
        return unary.Op switch
        {
            "-" => val is int i ? -i : val is long l ? -l : -ToDouble(val),
            "not" => !IsTruthy(val),
            _ => throw new InvalidOperationException($"Unknown unary operator: {unary.Op}"),
        };
    }

    private object? EvalFilter(FilterExpr filter, Scope scope)
    {
        var value = EvalExpr(filter.Value, scope);
        var args = filter.Args.Select(a => EvalExpr(a, scope)).ToArray();

        if (!_filters.TryGetValue(filter.Name, out var fn))
            throw new InvalidOperationException($"Unknown filter: {filter.Name}");

        return fn([value, .. args]);
    }

    private bool EvalTest(TestExpr test, Scope scope)
    {
        var value = EvalExpr(test.Value, scope);

        if (!_tests.TryGetValue(test.Name, out var fn))
            throw new InvalidOperationException($"Unknown test: {test.Name}");

        var result = fn(value, _globals);
        return test.Negated ? !result : result;
    }

    private object? EvalCall(CallExpr call, Scope scope)
    {
        var fn = EvalExpr(call.Function, scope);
        var args = call.Args.Select(a => EvalExpr(a, scope)).ToList();
        var kwargs = call.Kwargs.ToDictionary(kv => kv.Key, kv => EvalExpr(kv.Value, scope));

        if (fn is Delegate del)
            return del.DynamicInvoke([args.ToArray(), kwargs]);

        if (fn is string fnName && fnName == "raise_exception")
            throw new InvalidOperationException(args.FirstOrDefault()?.ToString() ?? "Template error");

        if (fn is Dictionary<string, object?> dict)
        {
            if (args.Count == 1 && args[0] is string key)
                return dict.TryGetValue(key, out var v) ? v : null;
        }

        return fn;
    }

    private static object? GetAttr(object? obj, string attr)
    {
        if (obj is null) return null;

        if (obj is Dictionary<string, object?> dict)
            return dict.TryGetValue(attr, out var val) ? val : null;

        return null;
    }

    private static object? GetItem(object? obj, object? key)
    {
        if (obj is null) return null;

        if (obj is Dictionary<string, object?> dict && key is string sKey)
            return dict.TryGetValue(sKey, out var val) ? val : null;

        var intKey = key switch
        {
            int i => i,
            double d when d == Math.Floor(d) => (int)d,
            long l => (int)l,
            _ => (int?)null,
        };

        if (obj is List<object?> list && intKey is int iIdx)
        {
            var idx = iIdx < 0 ? list.Count + iIdx : iIdx;
            return idx >= 0 && idx < list.Count ? list[idx] : null;
        }

        if (obj is object?[] arr && intKey is int aIdx)
        {
            var idx = aIdx < 0 ? arr.Length + aIdx : aIdx;
            return idx >= 0 && idx < arr.Length ? arr[idx] : null;
        }

        return null;
    }

    private static bool IsTruthy(object? val) => val switch
    {
        null => false,
        bool b => b,
        int i => i != 0,
        long l => l != 0,
        float f => f != 0,
        double d => d != 0,
        string s => s.Length > 0,
        List<object?> l => l.Count > 0,
        object?[] a => a.Length > 0,
        Dictionary<string, object?> d => d.Count > 0,
        _ => true,
    };

    private static double ToDouble(object? val) => val switch
    {
        null => 0,
        int i => i,
        long l => l,
        float f => f,
        double d => d,
        string s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : 0,
        bool b => b ? 1 : 0,
        _ => 0,
    };

    private static List<object?> ToList(object? val)
    {
        if (val is List<object?> list) return list;
        if (val is object?[] arr) return [.. arr];
        if (val is IEnumerable<object?> enumerable) return enumerable.ToList();
        return [];
    }

    private static int Compare(object? left, object? right)
    {
        if (left is null && right is null) return 0;
        if (left is null) return -1;
        if (right is null) return 1;

        if (left is string ls && right is string rs)
            return string.Compare(ls, rs, StringComparison.Ordinal);

        var ld = ToDouble(left);
        var rd = ToDouble(right);
        return ld.CompareTo(rd);
    }

    private static string Stringify(object? val) => val switch
    {
        null => "",
        bool b => b ? "true" : "false",
        _ => val.ToString() ?? "",
    };

    private static Dictionary<string, Func<object?[], object?>> CreateFilters() => new()
    {
        ["trim"] = args => args[0]?.ToString()?.Trim() ?? "",
        ["lower"] = args => args[0]?.ToString()?.ToLowerInvariant() ?? "",
        ["upper"] = args => args[0]?.ToString()?.ToUpperInvariant() ?? "",
        ["length"] = args => args[0] switch
        {
            List<object?> l => l.Count,
            object?[] a => a.Length,
            string s => s.Length,
            Dictionary<string, object?> d => d.Count,
            _ => 0,
        },
        ["join"] = args =>
        {
            var sep = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
            var items = args[0] switch
            {
                List<object?> l => l,
                object?[] a => a.ToList(),
                _ => new List<object?> { args[0] },
            };
            return string.Join(sep, items.Select(Stringify));
        },
        ["select"] = args =>
        {
            var list = ToList(args[0]);
            if (args.Length < 2) return list;
            var testName = args[1]?.ToString();
            return list.Where(item => testName switch
            {
                "defined" => item is not null,
                "string" => item is string,
                "number" => item is int or float or double or long,
                "mapping" => item is Dictionary<string, object?>,
                "iterable" => item is List<object?> or object?[],
                _ => true,
            }).ToList();
        },
        ["reject"] = args =>
        {
            var list = ToList(args[0]);
            if (args.Length < 3) return list;
            var testName = args[1]?.ToString();
            var testVal = args[2];
            return list.Where(item =>
            {
                if (testName == "equalto") return !Equals(item, testVal);
                return true;
            }).ToList();
        },
        ["first"] = args =>
        {
            var list = ToList(args[0]);
            return list.Count > 0 ? list[0] : null;
        },
        ["last"] = args =>
        {
            var list = ToList(args[0]);
            return list.Count > 0 ? list[^1] : null;
        },
        ["list"] = args => ToList(args[0]),
        ["items"] = args =>
        {
            if (args[0] is Dictionary<string, object?> dict)
                return dict.Select(kv => (object?)new List<object?> { kv.Key, kv.Value }).ToList();
            return new List<object?>();
        },
        ["string"] = args => Stringify(args[0]),
        ["tojson"] = args =>
        {
            var indent = args.Length > 1 ? args[1] as int? : null;
            return ToJson(args[0], indent);
        },
        ["default"] = args =>
        {
            var val = args[0];
            var defaultVal = args.Length > 1 ? args[1] : "";
            return IsTruthy(val) ? val : defaultVal;
        },
        ["d"] = args =>
        {
            var val = args[0];
            var defaultVal = args.Length > 1 ? args[1] : "";
            return IsTruthy(val) ? val : defaultVal;
        },
        ["replace"] = args =>
        {
            var s = args[0]?.ToString() ?? "";
            var old = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
            var @new = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
            return s.Replace(old, @new);
        },
    };

    private static Dictionary<string, Func<object?, Dictionary<string, object?>, bool>> CreateTests() => new()
    {
        ["defined"] = (val, _) => val is not null,
        ["undefined"] = (val, _) => val is null,
        ["none"] = (val, _) => val is null,
        ["string"] = (val, _) => val is string,
        ["number"] = (val, _) => val is int or float or double or long,
        ["mapping"] = (val, _) => val is Dictionary<string, object?>,
        ["iterable"] = (val, _) => val is List<object?> or object?[] or Dictionary<string, object?>,
        ["sequence"] = (val, _) => val is List<object?> or object?[],
        ["callable"] = (val, _) => val is Delegate,
        ["boolean"] = (val, _) => val is bool,
        ["true"] = (val, _) => IsTruthy(val),
        ["false"] = (val, _) => !IsTruthy(val),
        ["equalto"] = (val, globals) =>
        {
            return true;
        },
    };

    private static string ToJson(object? val, int? indent = null)
    {
        if (val is null) return "null";
        if (val is bool b) return b ? "true" : "false";
        if (val is int i) return i.ToString(CultureInfo.InvariantCulture);
        if (val is long l) return l.ToString(CultureInfo.InvariantCulture);
        if (val is float f) return f.ToString("R", CultureInfo.InvariantCulture);
        if (val is double d) return d.ToString("R", CultureInfo.InvariantCulture);
        if (val is string s) return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t") + "\"";

        if (val is List<object?> list)
        {
            var items = string.Join(", ", list.Select(v => ToJson(v, indent)));
            return $"[{items}]";
        }

        if (val is Dictionary<string, object?> dict)
        {
            var entries = string.Join(", ", dict.Select(kv => $"{ToJson(kv.Key, indent)}: {ToJson(kv.Value, indent)}"));
            return $"{{{entries}}}";
        }

        return val.ToString() ?? "null";
    }

    private sealed class Scope
    {
        private readonly Dictionary<string, object?> _vars;
        private readonly Scope? _parent;

        public Scope(Dictionary<string, object?> globals)
        {
            _vars = new Dictionary<string, object?>(globals);
        }

        private Scope(Scope parent)
        {
            _parent = parent;
            _vars = new Dictionary<string, object?>();
        }

        public object? this[string name]
        {
            get
            {
                if (_vars.TryGetValue(name, out var val)) return val;
                if (_parent is not null) return _parent[name];
                return null;
            }
            set => _vars[name] = value;
        }

        public bool TryGetValue(string name, out object? val)
        {
            if (_vars.TryGetValue(name, out val)) return true;
            if (_parent is not null) return _parent.TryGetValue(name, out val);
            val = null;
            return false;
        }

        public Scope Push() => new(this);
    }
}