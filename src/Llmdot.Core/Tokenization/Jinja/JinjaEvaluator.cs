using System.Globalization;
using System.Text;

namespace Llmdot.Tokenization.Jinja;

internal sealed class JinjaNamespace : Dictionary<string, object?>;

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
        _globals["raise_exception"] = (Func<object?[], Dictionary<string, object?>, object?>)((args, _) =>
            throw new InvalidOperationException(args.Length > 0 ? args[0]?.ToString() ?? "Template error" : "Template error"));
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
                if (setNode.Attr is not null)
                {
                    var ns = scope[setNode.Name];
                    if (ns is JinjaNamespace namespaceObj)
                        namespaceObj[setNode.Attr] = EvalExpr(setNode.Value, scope);
                    else if (ns is Dictionary<string, object?> dict)
                        dict[setNode.Attr] = EvalExpr(setNode.Value, scope);
                    else
                        scope[setNode.Name] = EvalExpr(setNode.Value, scope);
                }
                else
                {
                    scope[setNode.Name] = EvalExpr(setNode.Value, scope);
                }
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
            case SliceExpr slice:
                return EvalSlice(slice, scope);
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
            "in" => Contains(right, left),
            "not in" => !Contains(right, left),
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
        {
            try { return del.DynamicInvoke([args.ToArray(), kwargs]); }
            catch (System.Reflection.TargetInvocationException tie) when (tie.InnerException is not null)
            { throw tie.InnerException; }
        }

        if (fn is JinjaNamespace ns)
        {
            foreach (var kv in kwargs)
                ns[kv.Key] = kv.Value;
            return ns;
        }

        if (fn is Dictionary<string, object?> dict)
        {
            if (args.Count == 1 && args[0] is string key)
                return dict.TryGetValue(key, out var v) ? v : null;
        }

        if (fn is string fnName)
            return EvalMethodCall(fnName, args.Count > 0 ? args[0] : null, args.Skip(1).ToList(), kwargs);

        return fn;
    }

    private static object? EvalMethodCall(string methodName, object? obj, List<object?> args, Dictionary<string, object?> kwargs)
    {
        if (obj is string s)
        {
            if (methodName == "split")
            {
                var separator = args.Count > 0 ? args[0]?.ToString() ?? "" : null;
                var parts = separator is null ? s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries) : s.Split(separator, StringSplitOptions.None);
                return parts.Select(p => (object?)p).ToList();
            }
            if (methodName == "strip")
                return s.Trim();
            if (methodName == "lower")
                return s.ToLowerInvariant();
            if (methodName == "upper")
                return s.ToUpperInvariant();
            if (methodName == "startswith")
                return args.Count > 0 && s.StartsWith(args[0]?.ToString() ?? "", StringComparison.Ordinal);
            if (methodName == "endswith")
                return args.Count > 0 && s.EndsWith(args[0]?.ToString() ?? "", StringComparison.Ordinal);
            if (methodName == "replace")
            {
                var old = args.Count > 0 ? args[0]?.ToString() ?? "" : "";
                var @new = args.Count > 1 ? args[1]?.ToString() ?? "" : "";
                return s.Replace(old, @new);
            }
        }

        if (obj is List<object?> list)
        {
            if (methodName == "append" && args.Count > 0)
            {
                list.Add(args[0]);
                return null;
            }
        }

        return null;
    }

    private List<object?>? EvalSlice(SliceExpr slice, Scope scope)
    {
        var obj = EvalExpr(slice.Object, scope);
        var start = slice.Start is not null ? EvalExpr(slice.Start, scope) : null;
        var end = slice.End is not null ? EvalExpr(slice.End, scope) : null;

        var startIdx = start switch
        {
            int i => i,
            long l => (int)l,
            null => 0,
            _ => 0,
        };

        if (obj is List<object?> list)
        {
            var endIdx = end switch
            {
                int i => i,
                long l => (int)l,
                null => list.Count,
                _ => list.Count,
            };

            if (startIdx < 0) startIdx = list.Count + startIdx;
            if (endIdx < 0) endIdx = list.Count + endIdx;
            startIdx = Math.Max(0, Math.Min(startIdx, list.Count));
            endIdx = Math.Max(0, Math.Min(endIdx, list.Count));
            return list.Skip(startIdx).Take(endIdx - startIdx).ToList();
        }

        if (obj is object?[] arr)
        {
            var endIdx = end switch
            {
                int i => i,
                long l => (int)l,
                null => arr.Length,
                _ => arr.Length,
            };

            if (startIdx < 0) startIdx = arr.Length + startIdx;
            if (endIdx < 0) endIdx = arr.Length + endIdx;
            startIdx = Math.Max(0, Math.Min(startIdx, arr.Length));
            endIdx = Math.Max(0, Math.Min(endIdx, arr.Length));
            return arr[startIdx..endIdx].ToList();
        }

        return null;
    }

    private static bool Contains(object? container, object? item)
    {
        if (container is string s && item is string sub)
            return s.Contains(sub);
        if (container is List<object?> list)
            return list.Any(x => Equals(x, item));
        if (container is object?[] arr)
            return arr.Any(x => Equals(x, item));
        if (container is Dictionary<string, object?> dict && item is string key)
            return dict.ContainsKey(key);
        return false;
    }

    private static object? GetAttr(object? obj, string attr)
    {
        if (obj is null) return null;

        if (obj is Dictionary<string, object?> dict)
            return dict.TryGetValue(attr, out var val) ? val : null;

        if (obj is JinjaNamespace ns)
            return ns.TryGetValue(attr, out var val) ? val : null;

        if (obj is string s)
        {
            return attr switch
            {
                "split" => "split",
                "strip" => "strip",
                "lower" => "lower",
                "upper" => "upper",
                "startswith" => "startswith",
                "endswith" => "endswith",
                "replace" => "replace",
                _ => null,
            };
        }

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
            if (args.Length < 2) return list;
            var testName = args[1]?.ToString();
            var testVal = args.Length > 2 ? args[2] : null;
            return list.Where(item => testName switch
            {
                "equalto" => !Equals(item, testVal),
                "defined" => item is null,
                "undefined" => item is not null,
                "none" => item is not null,
                "string" => item is not string,
                "number" => item is not (int or float or double or long),
                "mapping" => item is not Dictionary<string, object?>,
                "iterable" => item is not (List<object?> or object?[]),
                "true" => !IsTruthy(item),
                "false" => IsTruthy(item),
                _ => true,
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
            return val is not null ? val : defaultVal;
        },
        ["d"] = args =>
        {
            var val = args[0];
            var defaultVal = args.Length > 1 ? args[1] : "";
            return val is not null ? val : defaultVal;
        },
        ["replace"] = args =>
        {
            var s = args[0]?.ToString() ?? "";
            var old = args.Length > 1 ? args[1]?.ToString() ?? "" : "";
            var @new = args.Length > 2 ? args[2]?.ToString() ?? "" : "";
            return s.Replace(old, @new);
        },
        ["map"] = args =>
        {
            var list = ToList(args[0]);
            if (args.Length > 1)
            {
                var arg = args[1]?.ToString();
                if (arg is not null)
                {
                    // Known filter names take priority over attribute lookup
                    var isFilter = arg is "upper" or "lower" or "trim" or "string" or "int" or "float"
                        or "title" or "capitalize" or "abs" or "round" or "length" or "tojson";
                    if (isFilter)
                    {
                        return (object?)list.Select(item => arg switch
                        {
                            "upper" => item?.ToString()?.ToUpperInvariant(),
                            "lower" => item?.ToString()?.ToLowerInvariant(),
                            "trim" => item?.ToString()?.Trim(),
                            "string" => Stringify(item),
                            "int" => (object?)(int)ToDouble(item),
                            "float" => (object?)ToDouble(item),
                            "title" => (object?)CultureInfo.InvariantCulture.TextInfo.ToTitleCase(item?.ToString()?.ToLowerInvariant() ?? ""),
                            "capitalize" => (object?)((item?.ToString() ?? "") is var cs && cs.Length > 0 ? char.ToUpperInvariant(cs[0]) + cs[1..].ToLowerInvariant() : cs),
                            "abs" => (object?)Math.Abs(ToDouble(item)),
                            "round" => (object?)Math.Round(ToDouble(item)),
                            "length" => (object?)ToList(item).Count,
                            "tojson" => (object?)ToJson(item),
                            _ => item,
                        }).ToList();
                    }
                    // Otherwise treat as attribute name
                    return (object?)list.Select(item => GetAttr(item, arg)).ToList();
                }
            }
            return (object?)list;
        },
        ["selectattr"] = args =>
        {
            var list = ToList(args[0]);
            if (args.Length < 2) return list;
            var attrName = args[1]?.ToString() ?? "";
            var testName = args.Length > 2 ? args[2]?.ToString() ?? "true" : "true";
            var testVal = args.Length > 3 ? args[3] : null;
            return list.Where(item =>
            {
                var attrVal = GetAttr(item, attrName);
                return testName switch
                {
                    "equalto" => Equals(attrVal, testVal),
                    "defined" => attrVal is not null,
                    "undefined" => attrVal is null,
                    "none" => attrVal is null,
                    "string" => attrVal is string,
                    "number" => attrVal is int or float or double or long,
                    "true" => IsTruthy(attrVal),
                    "false" => !IsTruthy(attrVal),
                    _ => IsTruthy(attrVal),
                };
            }).ToList();
        },
        ["rejectattr"] = args =>
        {
            var list = ToList(args[0]);
            if (args.Length < 2) return list;
            var attrName = args[1]?.ToString() ?? "";
            var testName = args.Length > 2 ? args[2]?.ToString() ?? "true" : "true";
            var testVal = args.Length > 3 ? args[3] : null;
            return list.Where(item =>
            {
                var attrVal = GetAttr(item, attrName);
                return testName switch
                {
                    "equalto" => !Equals(attrVal, testVal),
                    "defined" => attrVal is null,
                    "undefined" => attrVal is not null,
                    "none" => attrVal is not null,
                    "string" => attrVal is not string,
                    "number" => attrVal is not (int or float or double or long),
                    "true" => !IsTruthy(attrVal),
                    "false" => IsTruthy(attrVal),
                    _ => !IsTruthy(attrVal),
                };
            }).ToList();
        },
        ["unique"] = args =>
        {
            var list = ToList(args[0]);
            var seen = new HashSet<object?>(ObjectEqualityComparer.Instance);
            return list.Where(item => seen.Add(item)).ToList();
        },
        ["sort"] = args =>
        {
            var list = ToList(args[0]);
            var reverse = args.Length > 1 && IsTruthy(args[1]);
            var sorted = list.OrderBy(x => x, ObjectComparer.Instance).ToList();
            if (reverse) sorted.Reverse();
            return (object?)sorted;
        },
        ["reverse"] = args =>
        {
            var list = ToList(args[0]);
            var reversed = new List<object?>(list);
            reversed.Reverse();
            return (object?)reversed;
        },
        ["count"] = args => args[0] switch
        {
            List<object?> l => l.Count,
            object?[] a => a.Length,
            string s => s.Length,
            Dictionary<string, object?> d => d.Count,
            _ => 0,
        },
        ["int"] = args => (object?)(int)ToDouble(args[0]),
        ["float"] = args => (object?)ToDouble(args[0]),
        ["abs"] = args => (object?)Math.Abs(ToDouble(args[0])),
        ["round"] = args =>
        {
            var value = ToDouble(args[0]);
            var precision = args.Length > 1 ? (int)ToDouble(args[1]) : 0;
            return (object?)Math.Round(value, precision);
        },
        ["indent"] = args =>
        {
            var text = args[0]?.ToString() ?? "";
            var width = args.Length > 1 ? (int)ToDouble(args[1]) : 4;
            var indentFirst = args.Length > 2 && IsTruthy(args[2]);
            var pad = new string(' ', width);
            var lines = text.Split('\n');
            var sb = new StringBuilder();
            for (var i = 0; i < lines.Length; i++)
            {
                if (i > 0) sb.Append('\n');
                if ((i > 0 || indentFirst) && lines[i].Length > 0)
                    sb.Append(pad);
                sb.Append(lines[i]);
            }
            return (object?)sb.ToString();
        },
        ["title"] = args =>
        {
            var s = args[0]?.ToString() ?? "";
            return (object?)System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s);
        },
        ["capitalize"] = args =>
        {
            var s = args[0]?.ToString() ?? "";
            if (s.Length == 0) return (object?)s;
            return (object?)(char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant());
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
        ["equalto"] = (val, globals) => true,
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

    private sealed class ObjectEqualityComparer : IEqualityComparer<object?>
    {
        public static readonly ObjectEqualityComparer Instance = new();
        public new bool Equals(object? x, object? y) => JinjaEvaluator.Equals(x, y);
        public int GetHashCode(object? obj) => obj?.GetHashCode() ?? 0;
    }

    private sealed class ObjectComparer : IComparer<object?>
    {
        public static readonly ObjectComparer Instance = new();
        public int Compare(object? x, object? y) => JinjaEvaluator.Compare(x, y);
    }
}