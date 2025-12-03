"""
stats-compass-core: A clean toolkit of deterministic pandas-based data tools.

This package provides organized tools for data cleaning, transformation, EDA,
machine learning, and plotting, with automatic tool discovery and registration.
"""
from stats_compass_core.registry import ToolRegistry, registry

__version__ = "0.1.0"
__all__ = ["ToolRegistry", "registry"]

# Auto-discover and register all tools
registry.auto_discover()
