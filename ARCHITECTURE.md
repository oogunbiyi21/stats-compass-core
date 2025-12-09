# Architecture Guide

This document explains the architectural decisions in stats-compass-core, particularly the stateful design pattern that may seem unconventional for a Python library.

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

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Protocol Layer                      │
│  (JSON messages, tool schemas, LLM interaction)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tool Functions                         │
│  load_csv, drop_na, describe, train_linear_regression...    │
│  - Receive (state, params) tuple                            │
│  - Return Pydantic result models (JSON-serializable)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DataFrameState                          │
│  - Named DataFrame storage                                  │
│  - Model storage                                            │
│  - Memory management                                        │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### 1. All Tools Accept `state` Parameter

Every tool function follows this signature:

```python
def tool_name(state: DataFrameState, params: InputModel) -> ResultModel
```

This is intentional - it allows the MCP server to inject the shared state instance into each tool call.

### 2. Results are Always JSON-Serializable

We use Pydantic models for all results to guarantee JSON serializability:

- `DescribeResult` - summary statistics as dicts
- `ChartResult` - base64-encoded PNG images
- `ModelTrainingResult` - metrics and model ID (not the model itself)

### 3. Models Stored by ID, Not Returned

When you train a model:

```python
train_linear_regression(state, params)
# Returns: ModelTrainingResult(model_id="lr_abc123", metrics={...})
```

The actual sklearn model is stored in `state._models["lr_abc123"]`. The LLM receives only the ID, which it can use in subsequent calls:

```python
evaluate_classification_model(state, EvaluateInput(model_id="lr_abc123", ...))
```

### 4. Transforms Create New DataFrames (Copy-on-Write)

Transform operations like `pivot`, `groupby_aggregate`, and `filter_dataframe` create new DataFrames rather than mutating the source:

```python
# Source "sales" is not modified
pivot(state, PivotInput(dataframe_name="sales", result_name="sales_pivot", ...))
# Now have both "sales" (original) and "sales_pivot" (transformed)
```

This prevents unexpected side effects and allows the LLM to "undo" by simply using the original DataFrame name.

### 5. Optional Dependencies

Heavy dependencies (scikit-learn, matplotlib) are optional to keep the core lightweight:

```bash
pip install stats-compass-core        # Core only (pandas, numpy)
pip install stats-compass-core[ml]    # + scikit-learn
pip install stats-compass-core[plots] # + matplotlib, seaborn
pip install stats-compass-core[all]   # Everything
```

Tools gracefully fail with helpful messages when optional deps are missing.

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
