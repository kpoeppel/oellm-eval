"""What a launch planned, frozen beside it (<root>/launches/<id>/manifest.json; the counterpart of
the v0.02 protocol/manifest.json for plan launches). Workers add what actually ran
(<root>/provenance/, oellm.runner.provenance).

    plan       its sha256, model, protocol, task groups, oellm-eval revision, the rows done before
    units      per unit: kind, rows, backend, engine, env and the model directory its harness loads
    runtime    its sha256 and content; sha256 of its hash_files (patched harness files)
    code       the oellm-eval checkout and the caller's code identity (e.g. oellm-autoexp SYNC_STAMP)
    files      sha256 of every file of the launch directory
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from oellm.runner.plan import read_plan
from oellm.runner.provenance import code_identity, sha256
from oellm.runner.runtime import read_runtime


def model_of(plan, unit) -> str:
    """The model directory a unit's harness loads (commands.py): evalchemy the chat view, the others
    the plain model."""
    return plan.view_dir if unit.kind == "evalchemy" else plan.model_dir


def manifest(launch_dir: str | Path, plan_path: str | Path, runtime_path: str | Path, *,
             caller: dict | None = None) -> dict:
    launch_dir, plan_path, runtime_path = Path(launch_dir), Path(plan_path), Path(runtime_path)
    plan, runtime = read_plan(plan_path), read_runtime(runtime_path)
    return {
        "launch_id": launch_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path), "model_name": plan.model_name,
                 "model_dir": plan.model_dir, "view_dir": plan.view_dir, "root": plan.root, "limit": plan.limit,
                 "protocol": plan.protocol, "task_groups": plan.task_groups, "oellm_rev": plan.oellm_rev,
                 "rows": len(plan.rows), "rows_done_before": [r.name for r in plan.rows if plan.layout.is_done(r)],
                 "lm_eval_items": len(plan.lm_items)},
        "units": [{"id": u.id, "kind": u.kind, "rows": [r.name for r in u.rows], "backend": u.backend,
                   "engine": u.engine.__dict__, "env": u.env, "model": model_of(plan, u), "est_s": u.est_s}
                  for u in plan.units],
        "runtime": {"path": str(runtime_path), "sha256": sha256(runtime_path), "content": runtime.__dict__,
                    "hash_files": {f: sha256(f) for f in runtime.hash_files}},
        "code": {"oellm": code_identity(), "caller": caller or {}},
        "files": {str(f.relative_to(launch_dir)): sha256(f)
                  for f in sorted(launch_dir.rglob("*")) if f.is_file() and f.name != "manifest.json"},
    }


def write_manifest(launch_dir: str | Path, plan_path: str | Path, runtime_path: str | Path, *,
                   caller: dict | None = None) -> Path:
    path = Path(launch_dir) / "manifest.json"
    path.write_text(json.dumps(manifest(launch_dir, plan_path, runtime_path, caller=caller), indent=1) + "\n")
    return path
