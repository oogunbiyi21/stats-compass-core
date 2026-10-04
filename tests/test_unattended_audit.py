"""docs/audit/unattended.md must account for every subtool on the Stage A path.

The audit asks of every cleaning, transform, EDA and time-series subtool what it
does when nobody checks its output. The document is only useful while it is
complete, so this test holds it to the registry: a tool added without an audit
row fails here.
"""

import re
from pathlib import Path

from stats_compass_core.registry import registry

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "audit" / "unattended.md"
STATUSES = {"guarded", "warns", "not on path"}
ROW = re.compile(r"^\|\s*`(?P<tool>\w+)`\s*\|[^|]*\|\s*(?P<status>[a-z ]+?)\s*\|")


def _on_path_subtools() -> set[str]:
    tools = set()
    for category in ("cleaning", "transforms", "eda"):
        tools |= {
            t.name
            for t in registry.list_tools_by_tier(tiers=["sub"], category=category)
        }
    # Time series lives under ml beside the trainers, which are off the path.
    tools |= {
        t.name
        for t in registry.list_tools_by_tier(tiers=["sub"], category="ml")
        if "timeseries" in t.function.__module__
    }
    return tools


def _audited() -> dict[str, str]:
    rows = {}
    for line in AUDIT.read_text().splitlines():
        match = ROW.match(line)
        if match:
            rows[match["tool"]] = match["status"]
    return rows


def test_every_subtool_on_the_path_is_audited_and_nothing_else():
    audited = _audited()
    expected = _on_path_subtools()
    assert set(audited) == expected, {
        "missing from the audit": sorted(expected - set(audited)),
        "audited but not registered": sorted(set(audited) - expected),
    }
    assert len(audited) == len(expected)


def test_every_status_is_one_of_the_three():
    bad = {
        tool: status for tool, status in _audited().items() if status not in STATUSES
    }
    assert not bad, bad
