#!/usr/bin/env bash
# Does a core change leave the Shopify app's results unchanged?
#
# Runs hosted's commerce engine, analyze(), on its two demo inputs with core
# at BASE and at HEAD, and diffs the outputs byte for byte. Two fields are
# cleared first, and printed instead: each op's run time (ops[].ms), which
# differs between any two runs, and engineVersion, which names the core
# version and so changes with every release. A control runs BASE twice, to
# show nothing else varies.
#
# Usage: scripts/check_hosted_unchanged.sh HOSTED_REF CORE_BASE CORE_HEAD [WORKDIR]
#   e.g. scripts/check_hosted_unchanged.sh origin/main 54b0c87 HEAD /tmp/hosted-check
# Needs git, python3.12 and network access (it installs hosted's dependencies).
set -euo pipefail

HOSTED_REF=${1:?hosted ref, e.g. origin/main}
CORE_BASE=${2:?core base commit}
CORE_HEAD=${3:?core head commit}
WORK=${4:-$(mktemp -d)}
CORE_REPO=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
BASE_SHA=$(git -C "$CORE_REPO" rev-parse "$CORE_BASE")
HEAD_SHA=$(git -C "$CORE_REPO" rev-parse "$CORE_HEAD")
PYTHON=${PYTHON:-python3.12}

mkdir -p "$WORK"
if [ ! -d "$WORK/hosted" ]; then
  git clone -q https://github.com/oogunbiyi21/stats-compass-hosted.git "$WORK/hosted"
fi
git -C "$WORK/hosted" fetch -q origin
git -C "$WORK/hosted" checkout -q --detach "$HOSTED_REF"
HOSTED_SHA=$(git -C "$WORK/hosted" rev-parse HEAD)

for name in base head; do
  sha=$([ "$name" = base ] && echo "$BASE_SHA" || echo "$HEAD_SHA")
  rm -rf "$WORK/core-$name"
  mkdir -p "$WORK/core-$name"
  git -C "$CORE_REPO" archive "$sha" | tar -x -C "$WORK/core-$name"
done

if [ ! -x "$WORK/venv/bin/python" ]; then
  "$PYTHON" -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q --upgrade pip
  "$WORK/venv/bin/pip" install -q -e "$WORK/hosted"
fi

cat > "$WORK/run.py" <<'PY'
"""analyze() on hosted's demo inputs, as hosted serialises it; run times and the version label cleared."""
import gzip, sys
from pathlib import Path
from stats_compass_hosted.commerce.contracts import AnalysisInput
from stats_compass_hosted.commerce.engine import analyze
import stats_compass_core

fixtures = Path(sys.argv[1]) / "tests" / "commerce" / "fixtures"
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for name in ("analysis-input.demo.v1.json.gz", "analysis-input.demo-2026-03.v1.json.gz"):
    with gzip.open(fixtures / name, "rt") as f:
        result = analyze(AnalysisInput.model_validate_json(f.read()))
    print(f"{name}: engineVersion {result.engine_version}")
    result.engine_version = "cleared"
    for op in result.ops:
        op.ms = 0
    (out / name.replace(".json.gz", ".out.json")).write_text(result.model_dump_json(indent=1, by_alias=True))
print(f"core {stats_compass_core.__version__} from {Path(stats_compass_core.__file__).parent}")
PY

run() {
  "$WORK/venv/bin/pip" install -q --no-deps --force-reinstall -e "$WORK/core-$1"
  "$WORK/venv/bin/python" "$WORK/run.py" "$WORK/hosted" "$WORK/out-$2"
}
run base base
run base control
run head head

echo "hosted $HOSTED_SHA; core base $BASE_SHA; core head $HEAD_SHA"
echo "control (base twice):"
diff -r "$WORK/out-base" "$WORK/out-control" && echo "  identical"
echo "base against head:"
diff -r "$WORK/out-base" "$WORK/out-head" && echo "  identical"
for f in "$WORK"/out-head/*.json; do
  echo "  $(basename "$f"): $(wc -c < "$f") bytes, sha256 $(shasum -a 256 "$f" | cut -c1-16)"
done
