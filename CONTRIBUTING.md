# Contributing to Stats Compass Core

## Schema Design Guidelines

To ensure compatibility with strict MCP validators (like VS Code Copilot Chat), all tools must adhere to the following schema architecture:

### 1. Top-Level Inputs Must Be Strict
All tool input models must inherit from `StrictToolInput`. This enforces `additionalProperties: false` at the root level.

```python
from stats_compass_core.base import StrictToolInput

class MyToolInput(StrictToolInput):
    ...
```

### 2. Nested Components Must Be Permissive
Any nested object (referenced via `$ref`) must inherit from `ToolComponent`. This allows `additionalProperties` to be unspecified (or true), which prevents JSON Schema validation errors where strictness conflicts with reference definitions.

```python
from stats_compass_core.base import ToolComponent

class MySubComponent(ToolComponent):
    field: str
```

### 3. No `dict` Types
Avoid using `dict[str, Any]` or `dict[str, str]`. Instead, use a list of typed objects (inheriting from `ToolComponent`).

**Bad:**
```python
mappings: dict[str, str]
```

**Good:**
```python
class Mapping(ToolComponent):
    key: str
    value: str

mappings: list[Mapping]
```

## Supported Clients

Please note that we explicitly do **not** support the "Roo Code" extension due to its non-standard JSON Schema validation logic. All changes must be tested against official MCP clients (Claude Desktop, VS Code Copilot Chat).
