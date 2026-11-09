"""Which planned rows have no result in a collected CSV (``oellm-eval collect --check`` for worker runs).

A row (task, n_shot) counts as present when the CSV of collect.py has any metric for that task at
that n_shot (lm-eval groups count by their group row). The rows come from a plan, from task groups
or from a suite file (tasks: [{name, task, n_shot, suite}]):

    python -m oellm.reports.check --csv <model>.tasks.csv (--plan plan.json | --groups vllm=flag-evals-vllm,...
        | --suite flag_evals.yaml) [-o <model>.missing.csv] [--strict]

Prints a one-line count per harness; --strict exits 1 when a row is missing.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

from oellm.reports.collect import CSV_SPELLING


def missing_rows(rows: list[dict], csv_rows: list[dict]) -> list[dict]:
    """rows: dicts with name, task, n_shot, harness."""
    have = {(r["task"], str(r["n_shot"])) for r in csv_rows}
    return [t for t in rows if (CSV_SPELLING.get(t["task"], t["task"]), str(t["n_shot"])) not in have]


def planned_rows(*, plan: str | None = None, groups: str | None = None, suite: str | None = None) -> list[dict]:
    if plan:
        return [dict(r) for r in json.loads(Path(plan).read_text())["rows"]]
    if groups:
        from oellm.runner.plan import rows_from_task_groups

        halves = dict(kv.split("=", 1) for kv in groups.split(","))
        return [r.__dict__ for r in rows_from_task_groups(halves)]
    import yaml

    return [{"name": t["name"], "task": t["task"], "n_shot": t["n_shot"], "harness": t["suite"]}
            for t in yaml.safe_load(Path(suite).read_text())["tasks"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="collect.py output")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plan")
    src.add_argument("--groups")
    src.add_argument("--suite")
    ap.add_argument("-o", "--out", help="write the missing rows here")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any row is missing")
    a = ap.parse_args(argv)
    rows = planned_rows(plan=a.plan, groups=a.groups, suite=a.suite)
    with open(a.csv) as fh:
        miss = missing_rows(rows, list(csv.DictReader(fh)))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "task", "n_shot", "harness"])
            w.writeheader()
            w.writerows({k: t[k] for k in w.fieldnames} for t in miss)
    per = collections.Counter(t["harness"] for t in miss)
    total = collections.Counter(t["harness"] for t in rows)
    print(f"{Path(a.csv).name}: {len(rows) - len(miss)}/{len(rows)} rows present; missing "
          + (", ".join(f"{h} {per[h]}/{total[h]}" for h in sorted(per)) or "none"))
    for t in miss[:20]:
        print(f"  missing {t['name']}")
    if len(miss) > 20:
        print(f"  ... {len(miss) - 20} more" + (f" (in {a.out})" if a.out else ""))
    return 1 if a.strict and miss else 0


if __name__ == "__main__":
    sys.exit(main())
