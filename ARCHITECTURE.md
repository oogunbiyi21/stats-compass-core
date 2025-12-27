# Architecture Guide

This document explains the architectural decisions in stats-compass-core, particularly the stateful design pattern and workflow-based orchestration that may seem unconventional for a Python library.

## Why Server-Side State?

stats-compass-core is designed as a **Model Context Protocol (MCP) server**, not a traditional Python library. This distinction is crucial for understanding our design decisions.

### The MCP Constraint

In the MCP protocol, **LLMs cannot pass complex Python objects between tool calls**. All communication happens via JSON-serializable messages. This means:

- ❌ Cannot pass DataFrames directly between tools
- ❌ Cannot pass trained models between tools  
- ❌ Cannot maintain context across multiple operations
- ✅ Can pass simple identifiers (strings, numbers)
- ✅ Can pass JSON-serializable results

### The Traditional Alternative (and why it doesn't work)

A traditional functional approach might look like:

```python
# This CANNOT work in MCP - LLM cannot pass df between calls
df = load_csv("data.csv")
df = drop_na(df, columns=["age"])
result = describe(df)
```

The LLM would need to serialize the entire DataFrame in each message, which is:
1. Impractical (DataFrames can be gigabytes)
2. Slow (serialization overhead)
3. Error-prone (type conversion issues)

### Our Solution: Named References

Instead, we use a **named reference pattern**:

```python
# Tool calls via MCP
load_csv(path="data.csv")                    # → stores as "data"
drop_na(dataframe_name="data", columns=["age"])  # → updates in place
describe(dataframe_name="data")              # → returns JSON stats
```

The `DataFrameState` class manages:
- **Multi-DataFrame storage**: Multiple named DataFrames in memory
- **Active DataFrame tracking**: Default target when name not specified  
- **Model storage**: Trained ML models referenced by ID
- **Memory limits**: Prevents runaway memory usage (500MB default)

## Four-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Protocol Layer                      │
│  (JSON messages, tool schemas, LLM interaction)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Tools (Tier 1)                  │
│  run_preprocessing, run_classification, run_eda_report...   │
│  - Orchestrate multiple sub-tools                           │
│  - Return WorkflowResult with step-by-step details          │
│  - Handle complex multi-step operations                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Category Executors (Tier 2 - Optional)        │
│  execute_cleaning, execute_eda, describe_cleaning...        │
│  - Thin dispatchers to sub-tools                            │
│  - Reduce tool count for clients with limits                │
│  - Return sub-tool results unchanged                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Sub-Tool Functions (Tier 3)            │
│  load_csv, drop_na, describe, train_linear_regression...    │
│  - Receive (state, params) tuple                            │
│  - Return Pydantic result models (JSON-serializable)        │
│  - Atomic operations on DataFrames/models                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DataFrameState (Tier 4)                 │
│  - Named DataFrame storage                                  │
│  - Model storage                                            │
│  - Memory management                                        │
└─────────────────────────────────────────────────────────────┘
```

### Tier Breakdown

**Tier 1: Workflow Tools** - High-level orchestration for common tasks
- Single-call solutions for multi-step operations
- Example: `run_preprocessing` → analyze missing → impute → handle outliers → dedupe
- Returns `WorkflowResult` with step-by-step transparency

**Tier 2: Category Executors (Optional)** - Dynamic tool dispatchers
- `describe_cleaning` returns schemas for all cleaning sub-tools
- `execute_cleaning("drop_na", params)` dispatches to the actual sub-tool
- Used by MCP clients that struggle with 50+ tools (Gemini, GPT)
- Claude Desktop and VS Code work fine without this layer

**Tier 3: Sub-Tool Functions** - Atomic data operations
- Each does one thing well (drop rows, train model, create chart)
- All follow `(state, params) → result` signature
- Unchanged from original design - backward compatible

**Tier 4: DataFrameState** - Shared memory and resource management
- Multiple named DataFrames
- Trained model storage by ID
- Memory limits and cleanup

## Design Decisions

### 1. All Tools Accept `state` Parameter

Every tool function follows this signature:

```python
def tool_name(state: DataFrameState, params: InputModel) -> ResultModel
```

This is intentional - it allows the MCP server to inject the shared state instance into each tool call.

### 2. Workflow Tools Return Step-by-Step Results

Workflow tools orchestrate multiple sub-tools and return detailed execution results:

```python
@registry.register(category="workflows", tier="workflow")
def run_preprocessing(state: DataFrameState, params: RunPreprocessingInput) -> WorkflowResult:
    """Execute multiple cleaning steps in sequence."""
    steps = []
    
    # Step 1: Analyze missing data
    result1 = analyze_missing_data(state, AnalyzeMissingInput(...))
    steps.append(WorkflowStepResult(...))
    
    # Step 2: Clean dates (if configured)
    if params.config.date_cleaning:
        result2 = clean_dates(state, CleanDatesInput(...))
        steps.append(WorkflowStepResult(...))
    
    # Step 3: Apply imputation
    result3 = apply_imputation(state, ImputationInput(...))
    steps.append(WorkflowStepResult(...))
    
    # ... more steps ...
    
    return WorkflowResult(
        workflow_name="run_preprocessing",
        status="success",
        steps=steps,
        artifacts=WorkflowArtifacts(dataframes_created=[...], ...)
    )
```

The `WorkflowResult` provides:
- Transparency: What happened at each step
- Error handling: Which step failed and why
- Recovery hints: Suggestions for the LLM to retry
- Artifacts: Summary of created DataFrames, models, charts

### 3. Sub-Tools Remain Unchanged

The workflow layer is **purely additive**. Sub-tools retain their original signatures:

```python
def drop_na(state: DataFrameState, params: DropNAInput) -> DataFrameMutationResult
```

No changes to:
- Sub-tool function signatures
- Sub-tool input schemas (Pydantic models)
- Sub-tool result types
- Existing tests
- Existing notebooks

Workflows simply call sub-tools and aggregate their results.

### 4. Results are Always JSON-Serializable

We use Pydantic models for all results to guarantee JSON serializability:

- `DescribeResult` - summary statistics as dicts
- `ChartResult` - base64-encoded PNG images
- `ModelTrainingResult` - metrics and model ID (not the model itself)
- `WorkflowResult` - step-by-step execution details with nested results

### 5. Models Stored by ID, Not Returned

When you train a model:

```python
train_linear_regression(state, params)
# Returns: ModelTrainingResult(model_id="lr_abc123", metrics={...})
```

The actual sklearn model is stored in `state._models["lr_abc123"]`. The LLM receives only the ID, which it can use in subsequent calls:

```python
evaluate_classification_model(state, EvaluateInput(model_id="lr_abc123", ...))
```

### 6. Transforms Create New DataFrames (Copy-on-Write)

Transform operations like `pivot`, `groupby_aggregate`, and `filter_dataframe` create new DataFrames rather than mutating the source:

```python
# Source "sales" is not modified
pivot(state, PivotInput(dataframe_name="sales", result_name="sales_pivot", ...))
# Now have both "sales" (original) and "sales_pivot" (transformed)
```

This prevents unexpected side effects and allows the LLM to "undo" by simply using the original DataFrame name.

### 7. Optional Dependencies

Heavy dependencies (scikit-learn, matplotlib) are optional to keep the core lightweight:

```bash
pip install stats-compass-core        # Core only (pandas, numpy)
pip install stats-compass-core[ml]    # + scikit-learn
pip install stats-compass-core[plots] # + matplotlib, seaborn
pip install stats-compass-core[all]   # Everything
```

Tools gracefully fail with helpful messages when optional deps are missing.

## Workflow Architecture

### Workflow Data Structures

#### WorkflowStepResult
Each step in a workflow produces this:

```python
WorkflowStepResult
├── step_name: str                    # "load_csv", "describe", "impute"
├── step_index: int                   # 1, 2, 3...
├── status: StepStatus                # "success" | "failed" | "skipped"
├── duration_ms: int | None           # How long this step took
├── summary: str                      # Human-readable: "Loaded 1,234 rows"
├── result: dict | None               # The actual tool output (serialized)
├── image_base64: str | None          # If step produced a chart
├── error: str | None                 # If failed, what went wrong
├── skip_reason: str | None           # If skipped, why
└── dataframe_produced: str | None    # Name of any new DataFrame created
```

#### WorkflowResult
Returned by any workflow tool:

```python
WorkflowResult
├── workflow_name: str                # "run_eda_report"
├── status: WorkflowStatus            # "success" | "partial_failure" | "failed"
├── started_at: datetime
├── completed_at: datetime
├── total_duration_ms: int
├── input_dataframe: str              # What we started with
├── steps: list[WorkflowStepResult]   # All step results in order
├── artifacts: WorkflowArtifacts      # Summary of what was produced
├── error_summary: str | None         # If failed/partial, top-level explanation
├── suggestion: str | None            # Recovery hint for agent
└── recoverable: bool                 # Can agent retry with different params?
```

#### WorkflowArtifacts
Summary of everything produced:

```python
WorkflowArtifacts
├── dataframes_created: list[str]           # ["cleaned_data", "predictions"]
├── models_created: list[str]               # ["rf_classifier_churn_20241222"]
├── charts: list[ChartArtifact]             # Chart metadata and base64 images
└── final_dataframe: str | None             # The "main" output DataFrame
```

### Configuration Objects

Workflows accept optional config objects for customization:

```python
# Preprocessing with date cleaning
config = PreprocessingConfig(
    date_cleaning=DateCleaningConfig(
        date_column='Date',
        fill_method='ffill',
        infer_frequency=True
    ),
    imputation=ImputationConfig(strategy='median'),
    outliers=OutlierConfig(method='iqr', action='cap')
)
result = run_preprocessing(state, RunPreprocessingInput(config=config))
```

Available configs:
- `PreprocessingConfig` - imputation, outliers, date cleaning, dedupe
- `ClassificationConfig` - model type, hyperparameters, plots to generate
- `RegressionConfig` - model type, hyperparameters, plots to generate
- `EDAConfig` - which analyses/plots to include
- `TimeSeriesConfig` - ARIMA parameters, forecast periods, validation

### Available Workflows

#### EDA Report (`run_eda_report`)
Comprehensive data analysis in one call:
```
load_csv → get_schema → describe → analyze_missing_data 
→ correlations → data_quality_report
→ (auto-generate) histograms for numeric cols
→ (auto-generate) bar_charts for categorical cols
```
*Output: Complete summary + charts*

#### Preprocessing Pipeline (`run_preprocessing`)
Data cleaning and preparation:
```
analyze_missing_data → clean_dates (optional)
→ apply_imputation → handle_outliers 
→ dedupe
```
*Output: Cleaned DataFrame ready for ML*

#### Classification Pipeline (`run_classification`)
Train and evaluate classification model:
```
train_random_forest_classifier (or user choice)
→ evaluate_classification_model
→ confusion_matrix_plot
→ roc_curve_plot
→ precision_recall_curve_plot
→ feature_importance
```
*Output: Trained model + all evaluation artifacts*

#### Regression Pipeline (`run_regression`)
Train and evaluate regression model:
```
train_random_forest_regressor (or user choice)
→ evaluate_regression_model
→ feature_importance
→ predicted_vs_actual plot
```
*Output: Trained model + evaluation metrics*

#### Time Series Forecast (`run_timeseries_forecast`)
Complete ARIMA forecasting workflow:
```
validate_dates → check_stationarity
→ find_optimal_arima (if auto mode)
→ fit_arima
→ forecast_arima
→ arima_forecast_plot
```
*Output: Trained ARIMA model + forecast + visualization*

### Why Workflows?

Without workflows, agents need 8-10 tool calls for common tasks:
```
Agent: Load data
Agent: Check missing values  
Agent: Impute with mean
Agent: Check for outliers
Agent: Cap outliers
Agent: Remove duplicates
...
```

With workflows, one call does it all:
```
Agent: Run preprocessing on my data
→ [Workflow executes all steps]
→ Returns step-by-step results + final DataFrame
```

Benefits:
- **Reduced latency**: One round-trip instead of 10
- **Better UX**: Transparent multi-step execution
- **Error recovery**: Clear failure points with suggestions
- **Consistency**: Best practices baked in

## Memory Management

`DataFrameState` enforces a configurable memory limit (default 500MB):

```python
state = DataFrameState(max_memory_mb=1000)  # 1GB limit
```

When the limit is exceeded:
1. New DataFrame additions fail with `MemoryError`
2. Existing DataFrames remain accessible
3. LLM can delete DataFrames to free space

## Why Not Use a Traditional Design?

| Concern | Our Approach | Traditional Approach | Why We Chose Ours |
|---------|--------------|---------------------|-------------------|
| State persistence | Server-side `DataFrameState` | Return values | MCP can't pass DataFrames |
| Model storage | ID-based lookup | Return model object | Models aren't JSON-serializable |
| Charts | Base64 PNG strings | matplotlib Figure | Figures aren't JSON-serializable |
| Memory | Explicit limits | Unbounded | Prevent runaway server memory |

## Testing Considerations

When testing tools:

```python
# Always create fresh state for isolation
def test_my_tool():
    state = DataFrameState()
    state.set_dataframe("test", sample_df)
    
    result = my_tool(state, MyToolInput(...))
    
    assert result.success
    # State modifications are isolated to this test
```

## Further Reading

- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [README.md](./README.md) - Quick start and installation
- [API.md](./docs/API.md) - Complete tool reference
- [EXAMPLES.md](./docs/EXAMPLES.md) - Usage patterns
