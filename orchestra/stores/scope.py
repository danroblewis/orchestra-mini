"""PromQL's selectors, found and rewritten: ``walk`` replaces every vector selector of an expression
(``x``, ``x{a="b"}``, ``{__name__=~"a|b"}``) and leaves function and aggregation names, label lists
(``by (…)``, ``on (…)``), strings, durations and numbers as they are; ``matchers`` reads a
selector's braces. Copied from the old Orchestra's ``sources/scope.py``."""

from __future__ import annotations

import re

IDENT = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
LABEL = re.compile(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(=~|!~|!=|=)\s*")
AGGREGATIONS = {"sum", "avg", "count", "min", "max", "group", "stddev", "stdvar", "topk", "bottomk", "quantile",
                "count_values", "limitk", "limit_ratio"}
LISTS = {"by", "without", "on", "ignoring", "group_left", "group_right"}      # followed by a list of labels
WORDS = {"and", "or", "unless", "bool", "offset", "atan2", "inf", "nan"}


def _string(s: str, i: int) -> int:
    """The index just past the string literal starting at ``i``."""
    q, i = s[i], i + 1
    while i < len(s) and s[i] != q:
        i += 2 if s[i] == "\\" and q != "`" else 1
    return i + 1


def _close(s: str, i: int, opener: str, closer: str) -> int:
    """The index just past the bracket matching the one at ``i``, strings skipped."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c in "\"'`":
            i = _string(s, i)
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def matchers(block: str) -> list[tuple[str, str]]:
    """The matchers inside a selector's braces (without them): (label, matcher as written); a bare
    string (a quoted metric name) as ``("__name__", text)``."""
    out, i = [], 0
    while i < len(block):
        while i < len(block) and block[i] in " \t\n,":
            i += 1
        if i >= len(block):
            break
        if block[i] in "\"'`":
            j = _string(block, i)
            out.append(("__name__", block[i:j]))
            i = j
            continue
        m = LABEL.match(block, i)
        if not m:
            raise ValueError(f"not a label matcher: {block[i:]!r}")
        j = m.end()
        if j < len(block) and block[j] in "\"'`":
            j = _string(block, j)
        out.append((m[1], f"{m[1]}{m[2]}{block[m.end():j]}"))
        i = j
    return out


def _next(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def walk(expr: str, fn) -> str:
    """``expr`` with every vector selector replaced by ``fn(name, block, head)``: ``name`` the metric
    name written before the braces (or None), ``block`` the braces' content (None when there are
    none) and ``head`` the text before the braces as written (the name and any space after it).
    Function and aggregation names, label lists, strings, durations and numbers are left as they are."""
    out, i, n = [], 0, len(expr)
    while i < n:
        c = expr[i]
        if c in "\"'`":
            j = _string(expr, i)
            out.append(expr[i:j])
            i = j
        elif c == "[":
            j = _close(expr, i, "[", "]")
            out.append(expr[i:j])
            i = j
        elif c == "{":
            j = _close(expr, i, "{", "}")
            out.append(fn(None, expr[i + 1:j - 1], ""))
            i = j
        elif c.isdigit() or (c == "." and i + 1 < n and expr[i + 1].isdigit()):
            j = i
            while j < n and (expr[j].isalnum() or expr[j] in "._"):
                j += 1
            out.append(expr[i:j])
            i = j
        elif m := IDENT.match(expr, i):
            word, j = m[0], m.end()
            k = _next(expr, j)
            follows = expr[k] if k < n else ""
            if word.lower() in LISTS:
                out.append(word)
                if follows == "(":
                    e = _close(expr, k, "(", ")")
                    out.append(expr[j:e])
                    j = e
            elif follows == "(" or word.lower() in WORDS or (word in AGGREGATIONS and IDENT.match(expr, k)
                                                             and IDENT.match(expr, k)[0].lower() in LISTS):
                out.append(word)
            elif follows == "{":
                e = _close(expr, k, "{", "}")
                out.append(fn(word, expr[k + 1:e - 1], word + expr[j:k]))
                j = e
            else:
                out.append(fn(word, None, word))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)
