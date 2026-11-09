"""Plans from the command line (the caller then starts workers on its nodes).

    python -m oellm.runner.cli plan --model-dir <hf model> [--view-dir <identity view>] --out <dir> \\
        [--groups vllm=flag-evals-vllm,lighteval=flag-evals-lighteval] [--names n1,n2 ...] \\
        [--settings settings.json] [--limit 20] [--tag t] [--init-queue] -o plan.json
    python -m oellm.runner.cli status --plan plan.json
    python -m oellm.runner.cli requeue-failed --plan plan.json

settings.json: {"lm_eval": SessionSettings, "lighteval": GroupSettings, "evalchemy": GroupSettings}
(plan.py; a harness left out is not planned; ``{}`` takes the defaults). ``--names`` selects rows
of the groups by name.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

from oellm.runner.plan import (
    Engine, GroupSettings, SessionSettings, build_plan, init_queue, read_plan, rows_from_task_groups,
)
from oellm.runner.queue import Queue
from oellm.runner.runtime import OELLM_ROOT

DEFAULT_GROUPS = "vllm=flag-evals-vllm,lighteval=flag-evals-lighteval"


def _settings(cls, data: dict, where: str):
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    kw = dict(data)
    if "engine" in kw:
        bad = sorted(set(kw["engine"]) - {f.name for f in dataclasses.fields(Engine)})
        if bad:
            raise ValueError(f"{where}.engine: unknown keys {bad}")
        kw["engine"] = Engine(**kw["engine"])
    return cls(**kw)


def load_settings(path: str | None) -> dict:
    data = json.loads(Path(path).read_text()) if path else {"lm_eval": {}, "lighteval": {}, "evalchemy": {}}
    unknown = sorted(set(data) - {"lm_eval", "lighteval", "evalchemy"})
    if unknown:
        raise ValueError(f"settings: unknown keys {unknown}")
    return {"lm_eval": _settings(SessionSettings, data["lm_eval"], "lm_eval") if "lm_eval" in data else None,
            "lighteval": _settings(GroupSettings, data["lighteval"], "lighteval") if "lighteval" in data else None,
            "evalchemy": _settings(GroupSettings, data["evalchemy"], "evalchemy") if "evalchemy" in data else None}


def oellm_rev() -> str:
    try:
        rev = subprocess.run(["git", "-C", OELLM_ROOT, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", OELLM_ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return (rev or "unknown") + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--view-dir")
    p.add_argument("--model-name", help="default: the model directory's name")
    p.add_argument("--out", required=True, help="results root parent (<out>/<model name>[_limit<l>])")
    p.add_argument("--groups", default=DEFAULT_GROUPS, help="half=task group,...")
    p.add_argument("--names", help="comma-separated row names (a subset of the groups' rows)")
    p.add_argument("--harnesses", help="comma-separated harnesses to keep")
    p.add_argument("--settings")
    p.add_argument("--limit", type=float, default=0)
    p.add_argument("--protocol", default="v0.01")
    p.add_argument("--tag", default="")
    p.add_argument("--engine-times", help="JSON {row name: engine seconds} (lm-eval parts, balance)")
    p.add_argument("--row-times", help="JSON {harness|task|n_shot: job seconds} (group balance)")
    p.add_argument("--init-queue", action="store_true", help="queue the lm-eval items")
    p.add_argument("-o", "--output", required=True)
    for name in ("status", "requeue-failed"):
        s = sub.add_parser(name)
        s.add_argument("--plan", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        halves = dict(kv.split("=", 1) for kv in a.groups.split(","))
        rows = rows_from_task_groups(halves)
        if a.names:
            want = a.names.split(",")
            by = {r.name: r for r in rows}
            missing = [n for n in want if n not in by]
            if missing:
                raise SystemExit(f"rows not in the groups: {missing}")
            rows = [by[n] for n in dict.fromkeys(want)]
        if a.harnesses:
            keep = set(a.harnesses.split(","))
            rows = [r for r in rows if r.harness in keep]
        s = load_settings(a.settings)
        times = lambda f: json.loads(Path(f).read_text()) if f else {}  # noqa: E731
        model_dir = str(Path(a.model_dir).resolve())
        plan = build_plan(model_name=a.model_name or Path(model_dir).name, model_dir=model_dir,
                          view_dir=a.view_dir or model_dir, out=a.out, rows=rows, lm_eval=s["lm_eval"],
                          lighteval=s["lighteval"], evalchemy=s["evalchemy"], limit=a.limit, protocol=a.protocol,
                          task_groups=list(halves.values()), oellm_rev=oellm_rev(),
                          engine_times=times(a.engine_times), row_times=times(a.row_times), tag=a.tag)
        plan.write(a.output)
        print(f"{a.output}: {len(plan.units)} unit(s), {len(plan.rows)} row(s), {len(plan.lm_items)} lm-eval item(s)")
        for u in plan.units:
            print(f"  {u.id:10s} {u.kind:16s} {len(u.rows):3d} rows  est {u.est_s / 3600:5.1f} h")
        if a.init_queue and plan.lm_items:
            print(f"queue: {init_queue(plan)}")
        return 0
    plan = read_plan(a.plan)
    q = Queue(plan.layout.queue_dir)
    if a.cmd == "requeue-failed":
        print(f"requeued {q.requeue_failed()}; {q.status()}")
        return 0
    per: dict[str, list[int]] = {}
    for r in plan.rows:
        per.setdefault(r.harness, [0, 0])
        per[r.harness][1] += 1
        per[r.harness][0] += plan.layout.is_done(r)
    print(json.dumps({"rows_done": {h: f"{d}/{n}" for h, (d, n) in per.items()}, "queue": q.status(),
                      "units": [u.id for u in plan.units]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
