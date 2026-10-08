"""A small expression language for caller-supplied filters, columns and inspections.

filter_dataframe, add_column and inspect_data take text from whoever calls a
tool. On a shared server that is anyone who signs up, and on a laptop it can be
a model steered by text planted in a dataset. They used to hand that text to
pandas' own evaluator (``df.query`` / ``pd.eval``), which reaches attributes,
method calls, the calling function's locals and from there the file system
(security scan F1–F3, 8 Oct 2026).

This module parses the text with Python's ``ast`` and evaluates only what it
recognises. Anything else is refused before it runs:

- names: the frame's columns (bare, or in backticks when they are not
  identifiers), ``df`` for the whole frame, and ``index``;
- constants: numbers, strings, True/False/None, and lists or tuples of them;
- arithmetic ``+ - * / // % **``, comparisons (chained, ``in``/``not in``, and
  ``== [list]`` meaning "is one of", as in ``df.query``), ``and``/``or``/``not``
  and ``& | ~`` element by element;
- ``df["col"]``, ``df[["a", "b"]]`` and ``df[condition]``;
- the functions in ``FUNCTIONS`` (``np.log``, ``pd.to_numeric``, ``abs``...);
- the column and frame methods in ``SERIES_METHODS`` and ``FRAME_METHODS``,
  with constant arguments, and ``.str`` / ``.dt`` with the members listed.

``.str.contains`` matches literally: a caller-supplied regular expression can
backtrack for as long as it likes. Size is bounded too: the text's length, its
number of parts, the exponent of a power of two constants, and no repeating a
string, a list or a text column by multiplying it.
"""

from __future__ import annotations

import ast
import io
import operator
import tokenize
from typing import Any

import numpy as np
import pandas as pd

MAX_LENGTH = 2000
MAX_NODES = 300
MAX_SCALAR_EXPONENT = 64

_BACKTICK_PREFIX = "_sc_backtick_"


class ExpressionError(ValueError):
    """The expression is outside the language, or failed to evaluate."""


# Functions callable by name. The keys are what a caller writes; ``pd`` and
# ``np`` are never the modules themselves.
FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "len": len,
    "min": min,
    "max": max,
    "np.log": np.log,
    "np.log10": np.log10,
    "np.log2": np.log2,
    "np.log1p": np.log1p,
    "np.exp": np.exp,
    "np.sqrt": np.sqrt,
    "np.abs": np.abs,
    "np.round": np.round,
    "np.floor": np.floor,
    "np.ceil": np.ceil,
    "np.sign": np.sign,
    "np.where": np.where,
    "np.clip": np.clip,
    "np.maximum": np.maximum,
    "np.minimum": np.minimum,
    "np.isnan": np.isnan,
    "pd.to_numeric": pd.to_numeric,
    "pd.to_datetime": pd.to_datetime,
    "pd.isna": pd.isna,
    "pd.notna": pd.notna,
    "pd.isnull": pd.isnull,
    "pd.notnull": pd.notnull,
    "pd.Timestamp": pd.Timestamp,
}

# Keyword arguments a function in FUNCTIONS may take, all constants.
FUNCTION_KWARGS: dict[str, set[str]] = {
    "round": {"ndigits"},
    "np.round": {"decimals"},
    "np.clip": {"a_min", "a_max"},
    "pd.to_numeric": {"errors", "downcast"},
    "pd.to_datetime": {"errors", "format", "dayfirst", "yearfirst", "utc"},
}

CONSTANTS: dict[str, Any] = {
    "np.nan": np.nan,
    "np.inf": np.inf,
    "np.pi": np.pi,
    "np.e": np.e,
    "pd.NA": pd.NA,
    "pd.NaT": pd.NaT,
}

# Methods on a column. None of them takes a callable or touches a file; their
# arguments must be constants or columns.
SERIES_METHODS = {
    "isna", "notna", "isnull", "notnull", "isin", "between", "abs", "round",
    "fillna", "clip", "astype",
    "mean", "median", "sum", "min", "max", "std", "var", "count", "nunique",
    "quantile", "any", "all", "idxmin", "idxmax",
    "unique", "value_counts", "describe", "head", "tail", "tolist", "to_list",
}
FRAME_METHODS = {
    "mean", "median", "sum", "min", "max", "std", "var", "count", "nunique",
    "describe", "head", "tail", "isna", "notna",
}
FRAME_PROPERTIES = {"shape", "columns", "dtypes", "empty", "size"}
SERIES_PROPERTIES = {"dtype", "size", "empty", "name"}
STR_METHODS = {
    "contains", "startswith", "endswith", "lower", "upper", "strip", "lstrip",
    "rstrip", "len", "title", "isdigit", "isnumeric", "isalpha",
}
DT_PROPERTIES = {
    "year", "month", "day", "hour", "minute", "second", "dayofweek", "weekday",
    "dayofyear", "quarter", "date", "is_month_start", "is_month_end",
}
DT_METHODS = {"day_name", "month_name", "normalize"}
ASTYPE_TARGETS = {
    "int", "int32", "int64", "float", "float32", "float64", "str", "string",
    "bool", "category", "object", "datetime64[ns]",
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _Accessor:
    """``col.str`` or ``col.dt``: only the members listed above are reachable."""

    def __init__(self, kind: str, series: pd.Series) -> None:
        self.kind = kind
        self.series = series


def evaluate(expression: str, df: pd.DataFrame) -> Any:
    """Evaluate ``expression`` against ``df``. Raises ExpressionError if it cannot."""
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("The expression is empty.")
    if len(expression) > MAX_LENGTH:
        raise ExpressionError(f"The expression is longer than {MAX_LENGTH} characters.")
    if "@" in _outside_strings(expression):
        raise ExpressionError(
            "'@name' references are not supported; use column names and constants."
        )
    text, backticked = _replace_backticks(expression)
    try:
        tree = ast.parse(_replace_booleans(text).strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError, tokenize.TokenError) as exc:
        raise ExpressionError(f"Could not parse the expression: {exc}") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError(f"The expression has more than {MAX_NODES} parts.")
    try:
        return _Evaluator(df, backticked).visit(tree.body)
    except ExpressionError:
        raise
    except RecursionError:
        raise ExpressionError("The expression is nested too deeply.") from None
    except Exception as exc:  # pandas and numpy errors on the caller's data
        raise ExpressionError(f"{type(exc).__name__}: {exc}") from None


class _Evaluator:
    def __init__(self, df: pd.DataFrame, backticked: dict[str, str]) -> None:
        self.df = df
        self.backticked = backticked

    def visit(self, node: ast.AST) -> Any:
        handler = getattr(self, f"visit_{type(node).__name__}", None)
        if handler is None:
            raise ExpressionError(f"{_describe(node)} is not supported.")
        return handler(node)

    # -- leaves ------------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (bool, int, float, str)) or node.value is None:
            return node.value
        raise ExpressionError(f"Constant {node.value!r} is not supported.")

    def visit_Name(self, node: ast.Name) -> Any:
        name = self.backticked.get(node.id, node.id)
        if name in self.df.columns:
            return self.df[name]
        if node.id == "df":
            return self.df
        if node.id == "index":
            return pd.Series(self.df.index, index=self.df.index)
        if node.id in self.backticked:
            raise ExpressionError(f"There is no column named '{name}'.")
        raise ExpressionError(
            f"Unknown name '{name}'. Columns are: {', '.join(map(str, self.df.columns))}. "
            "Put a column name in backticks if it has spaces."
        )

    def visit_List(self, node: ast.List) -> list:
        return [self._constant(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self._constant(e) for e in node.elts)

    def _constant(self, node: ast.AST) -> Any:
        """A list element: a constant, or a negative one."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._constant(node.operand)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        raise ExpressionError("Lists may hold constants only.")

    # -- operators ---------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if isinstance(node.op, ast.MatMult):
            raise ExpressionError(
                "'@name' references are not supported; use column names and constants."
            )
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"Operator {type(node.op).__name__} is not supported.")
        left, right = self.visit(node.left), self.visit(node.right)
        for side in (left, right):
            if isinstance(side, (list, tuple, pd.DataFrame, _Accessor)):
                raise ExpressionError("Arithmetic works on columns and single values only.")
        if isinstance(node.op, ast.Mult) and (_is_text(left) or _is_text(right)):
            raise ExpressionError("Text cannot be repeated by multiplying it.")
        if isinstance(node.op, ast.Mod) and _is_text(left):
            raise ExpressionError("Text cannot be formatted with '%'.")
        if isinstance(node.op, ast.Pow) and _is_scalar(left) and _is_scalar(right):
            if not isinstance(right, (int, float)) or abs(right) > MAX_SCALAR_EXPONENT:
                raise ExpressionError(
                    f"A power of two constants may have an exponent up to {MAX_SCALAR_EXPONENT}."
                )
        if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            return op(_as_bool(left), _as_bool(right))
        return op(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(value, (list, tuple, pd.DataFrame, _Accessor)):
            raise ExpressionError("That operator works on columns and single values only.")
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, (ast.Not, ast.Invert)):
            value = _as_bool(value)
            return ~value if isinstance(value, (pd.Series, np.ndarray)) else not value
        raise ExpressionError(f"Operator {type(node.op).__name__} is not supported.")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [_as_bool(self.visit(v)) for v in node.values]
        combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        result = values[0]
        for value in values[1:]:
            result = combine(result, value)
        return result

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            step = self._compare(op, left, right)
            result = step if result is None else operator.and_(result, step)
            left = right
        return result

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        listed = isinstance(right, (list, tuple))
        if isinstance(op, (ast.In, ast.NotIn)) or (listed and isinstance(op, (ast.Eq, ast.NotEq))):
            if not listed and not isinstance(right, pd.Series):
                raise ExpressionError("'in' needs a list, e.g. region in ['US', 'UK'].")
            if isinstance(left, pd.Series):
                found = left.isin(list(right))
                return ~found if isinstance(op, (ast.NotIn, ast.NotEq)) else found
            found = left in list(right)
            return not found if isinstance(op, (ast.NotIn, ast.NotEq)) else found
        compare = _COMPARE.get(type(op))
        if compare is None:
            raise ExpressionError(f"Comparison {type(op).__name__} is not supported.")
        if isinstance(left, (pd.DataFrame, _Accessor)) or isinstance(right, (pd.DataFrame, _Accessor)):
            raise ExpressionError("Comparisons work on columns and single values only.")
        return compare(left, right)

    # -- selection ---------------------------------------------------------

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        key = self.visit(node.slice)
        if isinstance(value, pd.DataFrame):
            if isinstance(key, str):
                if key not in value.columns:
                    raise ExpressionError(f"There is no column named '{key}'.")
                return value[key]
            if isinstance(key, list) and all(isinstance(k, str) for k in key):
                missing = [k for k in key if k not in value.columns]
                if missing:
                    raise ExpressionError(f"There is no column named {missing[0]!r}.")
                return value[key]
            if _is_condition(key):
                return value[key]
        if isinstance(value, pd.Series) and _is_condition(key):
            return value[key]
        raise ExpressionError(
            "Selection supports df['col'], df[['a', 'b']] and a true/false condition."
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id in ("np", "pd"):
            qualified = f"{node.value.id}.{node.attr}"
            if qualified in CONSTANTS and node.value.id not in self._column_names():
                return CONSTANTS[qualified]
            raise ExpressionError(f"'{qualified}' is not available here.")
        value = self.visit(node.value)
        if isinstance(value, pd.Series):
            if node.attr in ("str", "dt"):
                return _Accessor(node.attr, value)
            if node.attr in SERIES_PROPERTIES:
                return getattr(value, node.attr)
        elif isinstance(value, pd.DataFrame):
            if node.attr in FRAME_PROPERTIES:
                return getattr(value, node.attr)
        elif isinstance(value, _Accessor) and value.kind == "dt" and node.attr in DT_PROPERTIES:
            return getattr(value.series.dt, node.attr)
        raise ExpressionError(f"'.{node.attr}' is not available here.")

    # -- calls -------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> Any:
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise ExpressionError("'*' and '**' arguments are not supported.")
        func = node.func
        if isinstance(func, ast.Name) and func.id in FUNCTIONS and func.id not in self._column_names():
            return self._call_function(func.id, node)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("np", "pd")
            and func.value.id not in self._column_names()
        ):
            qualified = f"{func.value.id}.{func.attr}"
            if qualified in FUNCTIONS:
                return self._call_function(qualified, node)
            raise ExpressionError(f"'{qualified}' is not available here.")
        if isinstance(func, ast.Attribute):
            return self._call_method(self.visit(func.value), func.attr, node)
        raise ExpressionError(f"{_describe(func)} cannot be called.")

    def _call_function(self, name: str, node: ast.Call) -> Any:
        args = [self.visit(a) for a in node.args]
        allowed = FUNCTION_KWARGS.get(name, set())
        kwargs = {}
        for k in node.keywords:
            if k.arg not in allowed:
                raise ExpressionError(f"{name}() does not take '{k.arg}' here.")
            kwargs[k.arg] = self._constant(k.value)
        for arg in args:
            if isinstance(arg, (pd.DataFrame, _Accessor)) and name != "len":
                raise ExpressionError(f"{name}() works on columns and single values.")
        return FUNCTIONS[name](*args, **kwargs)

    def _call_method(self, target: Any, method: str, node: ast.Call) -> Any:
        if isinstance(target, _Accessor):
            return self._call_accessor(target, method, node)
        if isinstance(target, pd.Series):
            allowed = SERIES_METHODS
        elif isinstance(target, pd.DataFrame):
            allowed = FRAME_METHODS
        else:
            raise ExpressionError(f"'.{method}()' is not available here.")
        if method not in allowed:
            raise ExpressionError(f"'.{method}()' is not available here.")
        args = [self._argument(a) for a in node.args]
        kwargs = {k.arg: self._argument(k.value) for k in node.keywords}
        if method == "astype":
            targets = args + list(kwargs.values())
            if len(targets) != 1 or targets[0] not in ASTYPE_TARGETS:
                raise ExpressionError(f"astype() takes one of: {', '.join(sorted(ASTYPE_TARGETS))}.")
        return getattr(target, method)(*args, **kwargs)

    def _call_accessor(self, accessor: _Accessor, method: str, node: ast.Call) -> Any:
        if accessor.kind == "str" and method in STR_METHODS:
            args = [self._constant(a) for a in node.args]
            kwargs = {k.arg: self._constant(k.value) for k in node.keywords}
            if method == "contains":
                if kwargs.get("regex"):
                    raise ExpressionError(".str.contains() matches text literally here.")
                kwargs["regex"] = False
            return getattr(accessor.series.str, method)(*args, **kwargs)
        if accessor.kind == "dt" and method in DT_METHODS:
            args = [self._constant(a) for a in node.args]
            return getattr(accessor.series.dt, method)(*args)
        raise ExpressionError(f"'.{accessor.kind}.{method}()' is not available here.")

    def _argument(self, node: ast.AST) -> Any:
        """A method argument: a constant, a list of constants, or a column."""
        value = self.visit(node)
        if isinstance(value, (pd.DataFrame, _Accessor)):
            raise ExpressionError("Method arguments may be constants, lists or columns.")
        return value

    def _column_names(self) -> set:
        return {c for c in self.df.columns if isinstance(c, str)}


# -- helpers ---------------------------------------------------------------


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, np.number)) and not isinstance(value, np.ndarray)


def _is_text(value: Any) -> bool:
    if isinstance(value, (str, bytes, list, tuple)):
        return True
    if isinstance(value, (pd.Series, np.ndarray)):
        return not pd.api.types.is_numeric_dtype(value.dtype) and not pd.api.types.is_bool_dtype(value.dtype)
    return False


def _is_condition(value: Any) -> bool:
    return isinstance(value, (pd.Series, np.ndarray)) and pd.api.types.is_bool_dtype(value.dtype)


def _as_bool(value: Any) -> Any:
    if isinstance(value, (list, tuple, pd.DataFrame, _Accessor)):
        raise ExpressionError("'and', 'or' and 'not' work on conditions.")
    return value


def _describe(node: ast.AST) -> str:
    names = {
        "Lambda": "A lambda",
        "ListComp": "A comprehension",
        "GeneratorExp": "A comprehension",
        "SetComp": "A comprehension",
        "DictComp": "A comprehension",
        "IfExp": "'x if c else y'",
        "JoinedStr": "An f-string",
        "NamedExpr": "':='",
        "Dict": "A dictionary",
        "Set": "A set",
        "Slice": "A slice",
    }
    return names.get(type(node).__name__, f"'{type(node).__name__}'")


def _outside_strings(text: str) -> str:
    """The text with every quoted string blanked out."""
    out, quote, escaped = [], None, False
    for ch in text:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            out.append(" ")
        elif ch in ("'", '"'):
            quote = ch
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _replace_booleans(text: str) -> str:
    """``&`` and ``|`` become ``and`` and ``or``, as pandas' own parser does.

    So ``price > 100 & region == 'US'`` means both conditions, as it does in
    ``df.query``, rather than Python's tighter-binding bitwise ``&``.
    """
    tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.OP and tok.string in ("&", "|"):
            tokens.append((tokenize.NAME, "and" if tok.string == "&" else "or"))
        else:
            tokens.append((tok.type, tok.string))
    return tokenize.untokenize(tokens)


def _replace_backticks(text: str) -> tuple[str, dict[str, str]]:
    """Swap each `column name` for an identifier, outside quoted strings."""
    names: dict[str, str] = {}
    out, quote, escaped, i = [], None, False, 0
    while i < len(text):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            out.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "`":
            end = text.find("`", i + 1)
            if end == -1:
                raise ExpressionError("A backtick is not closed.")
            placeholder = f"{_BACKTICK_PREFIX}{len(names)}"
            names[placeholder] = text[i + 1 : end]
            out.append(placeholder)
            i = end
        else:
            out.append(ch)
        i += 1
    return "".join(out), names


__all__ = ["ExpressionError", "evaluate", "MAX_LENGTH", "MAX_NODES"]
