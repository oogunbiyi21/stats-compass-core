# Security fixes for the 8 October 2026 scan

A Claude Security scan of revision `4afe4477` (0.1.38) found nine issues, all
from one root. The library takes free-form strings from whoever calls a tool and
passes them to an expression evaluator or to file reads and writes. That is
fine for one person on their own laptop. It is dangerous once the same tools
are served to many users from one process, as the Stats Compass MCP server
does: there the caller is anyone who signs up, or a model steered by text
planted in a dataset.

| Finding | Severity | Where | Fix | Commit |
|---|---|---|---|---|
| F1 | HIGH | `filter_dataframe` passed the query to `df.query` | own evaluator, `utils/safe_expr.py` | `948c574` |
| F2 | HIGH | `add_column` passed the expression to `pd.eval(engine="python")` behind a regex denylist | same evaluator | `948c574` |
| F3 | HIGH | `inspect_data`, as F2 | same evaluator | `948c574` |
| F4 | MEDIUM | trainers' `save_path` reached `joblib.dump` directly | `safe_save` with the write root | `f2a0281` |
| F5 | MEDIUM | `fit_arima`'s `save_path`, as F4 | `safe_save` with the write root | `f2a0281` |
| F6 | MEDIUM | plot `save_path` written anywhere outside a few system folders | `FilePolicy.write_root` | `bfc01ed` |
| F7 | MEDIUM | workflow `model_save_path`, and a `save_path` hyperparameter even with `save_model=False` | reserved keys refused; `safe_save` | `f2a0281` |
| F8 | MEDIUM | `load_csv`, `load_excel`, `list_files` read any path | `FilePolicy.read_roots` | `bfc01ed` |
| F9 | LOW | CSV sanitiser skipped headers, row labels, `string` and `category` columns | all of them sanitised | `6288598` |

Found on the way:
- Fixed with F8: `load_dataset` built its path from the dataset name, so
  `../../x` read any `.csv` on disk. A name must now match
  `[A-Za-z0-9][A-Za-z0-9_-]*` and resolve inside the datasets folder.
- `groupby_aggregate` defined `VALID_AGGS` but never checked it, and `pivot`'s
  `aggfunc` took any string. pandas calls a group's method by the name it is
  given, so `plot` ran. Both schemas now accept only the listed names
  (`AggregationName`), which also shows them to clients.

## 1. Expressions are read, not executed

`filter_dataframe`, `add_column` and `inspect_data` share one evaluator,
`stats_compass_core.utils.safe_expr.evaluate`. It parses the text with Python's
`ast` and evaluates only what it recognises. pandas' evaluator is never called
with caller text; `tests/test_safe_expressions.py` spies on `pd.eval`,
`DataFrame.query` and `DataFrame.eval` and fails if any is reached.

**Allowed:**
- column names, bare or in backticks;
- `df` for selection: `df["col"]`, `df[["a", "b"]]`, `df[condition]`;
- `index`;
- constants, and lists or tuples of them;
- `+ - * / // % **`;
- comparisons, chained or not, with `in` / `not in`, and `== [list]` meaning
  "is one of";
- `and`, `or`, `not`, `&`, `|`, `~`, all element by element.

`&` and `|` are rewritten to `and` and `or` before parsing, as pandas' own parser
does, so `price > 100 & region == 'US'` means both conditions, as it did.

**Functions** (`FUNCTIONS`): `abs`, `round`, `len`, `min`, `max`, and these:

| Prefix | Functions |
|---|---|
| `np.` | `log`, `log10`, `log2`, `log1p`, `exp`, `sqrt`, `abs`, `round`, `floor`, `ceil`, `sign`, `where`, `clip`, `maximum`, `minimum`, `isnan` |
| `pd.` | `to_numeric`, `to_datetime`, `isna`, `notna`, `isnull`, `notnull`, `Timestamp` |

Keyword arguments are allowed only where `FUNCTION_KWARGS` names them, and must
be constants. `np` and `pd` are table prefixes, never the modules.

**Methods**, with constant or column arguments. Each method lists how many
positional arguments it takes and which keywords (`METHOD_ARGS`, `STR_ARGS`,
`DT_ARGS`). Listing the arguments matters as much as listing the methods:
`value_counts(bins=10**9)`, or the same `bins` passed by position, asks pandas
for a billion bins.

| On | Methods |
|---|---|
| Columns (`SERIES_METHODS`) | `isna`, `notna`, `isin`, `between`, `abs`, `round`, `fillna`, `clip`, `astype` (to a fixed list of types) |
| Columns: reductions | `mean`, `median`, `sum`, `min`, `max`, `std`, `var`, `count`, `nunique`, `quantile`, `any`, `all`, `idxmin`, `idxmax` |
| Columns: inspection | `unique`, `value_counts`, `describe`, `head`, `tail`, `tolist` |
| Frames | the reductions, plus `describe`, `head`, `tail`, `isna`, `notna` |

**Accessors:**
- `.str`: `contains`, `startswith`, `endswith`, `lower`, `upper`, `strip`,
  `len`. `contains` matches literally (`regex=False`), because a caller-supplied
  regular expression can backtrack without limit.
- `.dt`: `year`, `month`, `day`, `hour`, `dayofweek`, `quarter` and similar
  properties, plus `day_name()`, `month_name()`, `normalize()`.

**Refused**, as `ValueError` before anything runs:
- every other attribute or method, including file writers, `eval` and `query`;
- `@` references;
- lambdas, comprehensions and f-strings;
- calls on anything but the tables above;
- `*args` and `**kwargs`.

**Bounded:**
- 2,000 characters and 300 syntax nodes;
- a power of two constants may have an exponent up to 64, and a whole-number
  result is capped at 65,536 bits. The exponent cap alone does not bound the
  base: `((2**64)**64)**64` keeps every exponent at 64;
- positional and keyword arguments per function and method, as listed;
- `round` takes up to 15 decimals. A billion took over a second;
- no multiplying a string, a list or a text column, and no `%` formatting of
  text. Each of these can allocate gigabytes from a few characters;
- no powers on a column that is not numeric, and `astype("object")` is not a
  cast target. An object column, whether cast or loaded from mixed data, does
  integer arithmetic with Python's unbounded integers, so
  `2 ** (col + 10**9)` built a 125 MB number per row (pre-release review F1,
  8 Oct 2026). A numeric column cannot do this: pandas refuses Python integers
  too large for it.

**Behaviour changes for callers:**
- `@name` references, `pd.`/`np.` functions outside the table, and arbitrary
  methods no longer work. That includes `groupby` in `inspect_data`; use
  `groupby_aggregate`.
- `.str.contains` is literal.
- `inspect_data` now accepts `==`. The old guard refused any `=`, so it could
  not compare at all.
- `add_column`'s documented example `np.log(price)` and `np.where(...)` now work;
  under `pd.eval` they failed (found by dev-95).

## 2. Where tools may read and write: `FilePolicy`

`DataFrameState(file_policy=FilePolicy(write_root=..., read_roots=(...)))`. It
can be assigned later as `state.file_policy`. Without one, the state reads
`STATS_COMPASS_WRITE_ROOT` and `STATS_COMPASS_READ_ROOTS` (separated by
`os.pathsep`). With neither, it is unconfined, which is the laptop default: a
user's own paths keep working.

**Under a write root:**
- Every write lands inside the root: `save_csv`, `save_model`, plot
  `save_path`, trainer and ARIMA `save_path`, and workflow model files.
  - A relative path is taken relative to the root.
  - A path that then lies inside the root keeps its folders, so a server's own
    layout (`data/`, `models/`) survives.
  - Anything that would land outside, through an absolute path, `..` or a
    symlinked folder, is written to the root under its base name instead.
  - The first version of this branch used the base name for everything; the
    MCP server's per-category export folders needed the change. Only tests
    added on this branch changed with it.
- The extension must fit the file:
  - CSV: `.csv`, `.tsv`, `.txt`;
  - model: `.joblib`, `.pkl`, `.pickle`;
  - figure: `.png`, `.svg`, `.pdf`, `.jpg`, `.jpeg`.
- The root is created if missing.
- Containment is checked on the real path, and a symlink planted in the root,
  even a dangling one, is stepped round rather than written through.

**Under read roots:**
- `load_csv`, `load_excel` and `list_files` must resolve, after symlinks and
  `..`, inside one of the roots.
- A relative path is taken relative to the first root.

**Always:**
- Writes go through `safe_write_path`, which never overwrites: an existing name
  gets `_1`, `_2` and so on.
- Protected extensions (`.py`, `.sh`, `.toml` and others) are refused.
- The system-folder denylist checks both the plain and the real path against both
  forms of each folder. On macOS `/etc` is a link to `/private/etc`.
- `UnsafePathError` is now a `ValueError`.

## 3. Model files

- Every model write goes through `safe_save`.
  `tests/test_model_writes.py::test_only_file_safety_dumps_models` fails if
  `joblib.dump(` appears anywhere else in the package.
- `save_model=False` writes nothing.
- `build_training_params` refuses hyperparameters naming `dataframe_name`,
  `target_column`, `feature_columns` or `save_path`, the keys the workflow sets
  itself. The train step fails with a message saying so. Other hyperparameters
  still override the common parameters as before.

**Behaviour changes:**
- Saving a model to an existing file name now writes `name_1.joblib` instead of
  overwriting.
- A model path with a protected extension is refused.
- Under a write root, a workflow's default model file lands in the root, not in
  the system temp folder.

## 4. CSV exports

`sanitize_dataframe` now prefixes a quote to every string that starts with `=`,
`+`, `-`, `@`, a tab or a line break. That covers cells in object, `string` and
`category` columns, column headers (including each level of a two-level header),
row labels and index names.

A `category` column comes back as object, because quoting can merge or split
categories. The input frame is never modified.

## 5. What a server must do

Core's confinement is **off unless the server turns it on**. On a server
serving more than one user:

- Set a `FilePolicy` on each session's state: `write_root` = that session's
  export folder, `read_roots` = that session's upload folder. Alternatively, set
  the two environment variables for a single-tenant process.
- The scan's report says the MCP layer already confines `save_csv` and
  `save_model` writes. It does not: stats-compass-mcp 0.3.31 passes paths
  straight through. Its network mode must set the policy, or this switch does
  nothing there.

## 5a. Re-scan of 0.1.39, 9 October 2026

A full re-scan of `main` at `eb3391f` found eight issues: six MEDIUM and two
LOW. Fixed for 0.1.40.

| Finding | Where | Fix |
|---|---|---|
| F1–F5, F7 | `histogram`, `lineplot`, `scatter_plot`, `bar_chart`, `confusion_matrix_plot` and `feature_importance` called `safe_save` without the session's write root | see below |
| F6 | `run_step` called `summary_template.format(result=...)` on text that already held column names | the summary is used exactly as given |
| F8 | `describe`'s `include`/`exclude` were free strings reaching pandas' dtype parsing | a fixed list of names |

**F1–F5, F7: the plot tools ignored the write root.**
- The first pass checked these tools with a search whose output was cut off
  after 40 lines. Six plot tools were past the cut, so in a confined session a
  caller's `save_path` still wrote figures anywhere writable.
- Hosted's guard blocked this; self-hosted `serve` did not.
- All six now pass `root=state.file_policy.write_root`.
- An omitted `root` no longer means "anywhere": `safe_save`, `safe_write_path`
  and `safe_save_figure` default to `FROM_ENVIRONMENT`, which is
  `STATS_COMPASS_WRITE_ROOT`, or unconfined when that is unset. Passing
  `root=None` is a decision.
- `tests/test_every_writer_is_confined.py` keeps it that way:
  - it calls every registered tool with a `save_path` or `filepath` field under
    a write root, and checks the file lands inside;
  - it fails if a new writer is not in its table;
  - it checks that every save call in the package names its root.

**F6: column names became format strings.**
- Callers build summaries with column names and the date column in them, and
  those come from data: CSV and Sheet headers, `rename_columns`,
  `date_column`.
- Braces in a header became format fields. A field that walks attributes to a
  plain string takes a width; dev-95 measured 100 MB in 0.19 s from one
  histogram step on production.
- No template used `{result}`, so `run_step` no longer formats at all.

**F8: describe's type names backtracked.**
- `describe` now accepts only `all`, `number`, `object`, `category`,
  `datetime`, `bool`, `string` and `timedelta`, or a list of up to eight.
- `.describe()` in the expression language takes the same names.
- The free string reached pandas' dtype parsing, whose datetime pattern
  backtracks quadratically on `M8[` followed by `, ` repeated.

## 6. Not covered

- **Size of the data.** A long computation over a large frame is bounded by the
  data, not the expression. The state's memory limit applies.
- **A race between the containment check and the write.** Someone with write
  access to the root at the same moment could plant a symlink in between.
- **Unconfined reads on a laptop.** They are by design: that is the user's own
  machine.
