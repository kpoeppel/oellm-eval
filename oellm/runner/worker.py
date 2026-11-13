"""Run units of a plan on the GPUs of the current allocation.

    python -m oellm.runner.worker --plan plan.json --runtime runtime.json --unit le2
    python -m oellm.runner.worker --plan plan.json --runtime runtime.json --index $SLURM_ARRAY_TASK_ID
    python -m oellm.runner.worker --plan plan.json --runtime runtime.json --all   # one after another
    ... --print                                                                    # commands only

A lighteval / evalchemy unit runs its rows without a done marker and then marks every row whose
result is there (a failed call leaves its rows unmarked: a relaunch reruns them). An lm-eval
session unit runs the queue's items until the queue is empty (``replicas`` engines, each on its
share of the node's GPUs). Every command runs in its own process group, its output mirrored; one
that stays silent ``hang_minutes`` after vLLM's shutdown lines is ended (the post-shutdown hang).
SIGTERM ends the running commands and the worker (exit 143). A provenance record per unit run goes
to <root>/provenance/ (provenance.py).

Exit status: 0 when the unit's rows are done (or a session ended with the queue empty), 1 when rows
are left without a result (0 with --missing-ok) or a session failed, 2 for a bad plan / runtime.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from oellm.runner import provenance
from oellm.runner.commands import Command, unit_commands
from oellm.runner.layout import Row
from oellm.runner.plan import Plan, Unit, read_plan
from oellm.runner.runtime import Runtime, read_runtime

SHUTDOWN_MARK = "[shutdown]"
_children: list[subprocess.Popen] = []


# ---------------------------------------------------------------- results -> done markers


def result_keys(results_dir: Path, harness: str) -> set[str]:
    """Task keys with a result: evalchemy ``<task>``, lighteval ``<task>|<n_shot>``."""
    root = results_dir / "results" if harness == "lighteval" else results_dir
    keys: set[str] = set()
    for path in root.glob("**/results_*.json"):
        try:
            keys.update((json.loads(path.read_text()).get("results") or {}).keys())
        except (OSError, ValueError):
            continue
    return keys


def mark_done(plan: Plan, cmd: Command) -> tuple[list[Row], list[Row]]:
    """Done markers for the command's rows that have a result: (marked, without a result)."""
    layout = plan.layout
    out = Path(cmd.argv[cmd.argv.index("--output-dir" if cmd.harness == "lighteval" else "--output_path") + 1])
    keys = result_keys(out, cmd.harness)
    done, missing = [], []
    for r in cmd.rows:
        ok = (f"{r.task}|{r.n_shot}" in keys) if cmd.harness == "lighteval" else (r.task in keys)
        if not ok:
            missing.append(r)
            continue
        marker = layout.marker(r)
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps({"task": r.task, "n_shot": r.n_shot, "suite": r.harness, "limit": plan.limit,
                                          "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": str(out)}) + "\n")
        done.append(r)
    return done, missing


# ---------------------------------------------------------------- processes


def _mounts(plan: Plan) -> list[str]:
    return [p for p in dict.fromkeys([plan.model_dir, plan.view_dir, plan.root]) if p]


def start(runtime: Runtime, plan: Plan, cmd: Command, extra_env: dict[str, str] | None = None):
    argv, env, cwd = runtime.wrap(cmd.argv, cmd.harness, {**cmd.env, **(extra_env or {})}, cmd.cwd, _mounts(plan))
    print(f"[worker] start: {' '.join(argv[:12])}{' ...' if len(argv) > 12 else ''}", flush=True)
    proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, bufsize=1, text=True,
                            errors="replace")
    _children.append(proc)
    return proc, argv


def wait(proc: subprocess.Popen, hang_minutes: float, prefix: str = "") -> tuple[int, bool]:
    """Mirror the output until the process ends; (exit status, ended by the hang watchdog)."""
    state = {"last": time.time(), "shutdown": False}

    def pump():
        for line in proc.stdout:
            state["last"] = time.time()
            if SHUTDOWN_MARK in line:
                state["shutdown"] = True
            sys.stdout.write(prefix + line)
        sys.stdout.flush()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    hung = False
    while proc.poll() is None:
        time.sleep(1)
        if hang_minutes and state["shutdown"] and time.time() - state["last"] > hang_minutes * 60:
            print(f"[worker] no output for {hang_minutes:g} min after vLLM's shutdown: ending the command", flush=True)
            _kill(proc)
            hung = True
            break
    proc.wait()
    t.join(timeout=10)
    if proc in _children:
        _children.remove(proc)
    return (124 if hung else proc.returncode), hung


def _kill(proc: subprocess.Popen, grace: float = 30.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _on_term(*_):
    for p in list(_children):
        _kill(p, grace=10)
    sys.exit(143)


# ---------------------------------------------------------------- units


def replica_gpus(runtime: Runtime, unit: Unit) -> list[str | None]:
    """CUDA_VISIBLE_DEVICES per replica (None: leave it as it is)."""
    n = unit.engine.replicas
    if n == 1:
        return [None]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(runtime.gpus)]
    per = unit.engine.tp * unit.engine.dp
    if per * n > len(devices):
        raise ValueError(f"unit {unit.id}: {n} replicas x {per} GPUs > {len(devices)} GPUs")
    return [",".join(devices[i * per:(i + 1) * per]) for i in range(n)]


def run_unit(plan: Plan, runtime: Runtime, unit: Unit, *, plan_path: str = "", runtime_path: str = "",
             probe: bool = True, dry: bool = False, missing_ok: bool = False) -> int:
    layout = plan.layout
    rows = [r for r in unit.rows if not layout.is_done(r)]
    shard0 = [u.id for u in plan.units].index(unit.id) * 10
    cmds = unit_commands(plan, unit, rows, shard=shard0)
    if dry:
        for c in cmds:
            argv, _, cwd = runtime.wrap(c.argv, c.harness, c.env, c.cwd, _mounts(plan))
            print(json.dumps({"argv": argv, "cwd": cwd, "env": c.env}))
        return 0
    if not cmds:
        print(f"[worker] {unit.id}: every row is done", flush=True)
        return 0
    record = {"plan": plan_path, "plan_sha256": provenance.sha256(plan_path) if plan_path else None,
              "runtime": {k: v for k, v in runtime.__dict__.items() if k != "active_tmp"}, "runtime_path": runtime_path,
              "runtime_sha256": provenance.sha256(runtime_path) if runtime_path else None,
              "unit": {"id": unit.id, "kind": unit.kind, "rows": [r.name for r in unit.rows], "backend": unit.backend,
                       "engine": unit.engine.__dict__, "env": unit.env},
              "model": {"name": plan.model_name, "dir": plan.model_dir, "view": plan.view_dir},
              "protocol": plan.protocol, "limit": plan.limit, "start_utc": provenance.now(),
              "code": provenance.code_identity(), "machine": provenance.machine(),
              "hash_files": {f: provenance.sha256(f) for f in runtime.hash_files}, "commands": []}
    if probe or runtime.require:
        record["harnesses"] = {h: provenance.probe_versions(runtime, h) for h in dict.fromkeys(c.harness for c in cmds)}
        bad = provenance.version_mismatches(runtime.require, record["harnesses"])
        if bad:
            # a different harness version changes prompts or metrics: no row of this unit runs
            print(f"[worker] {unit.id}: runtime differs from the pinned versions: " + "; ".join(bad), flush=True)
            record.update(end_utc=provenance.now(), rows={"done": [], "missing": [r.name for r in rows]}, rc=3,
                          version_mismatch=bad)
            path = provenance.write_record(layout.provenance_dir, unit.id, record)
            print(f"[worker] {unit.id}: rc 3, provenance {path}", flush=True)
            return 3
    tmp = runtime.open_tmp()
    if tmp:
        print(f"[worker] {unit.id}: private /tmp {tmp}", flush=True)
    removed = runtime.cleanup()
    if removed:
        print(f"[worker] removed {len(removed)} leftover file(s): {removed[:4]}", flush=True)
    for h in dict.fromkeys(c.harness for c in cmds):
        runtime.prune_for(h)
    rc = 0
    done: list[Row] = []
    missing: list[Row] = []
    try:
        if unit.kind == "lm_eval_session":
            from oellm.runner.queue import Queue

            q = Queue(layout.queue_dir)
            print(f"[worker] {unit.id}: lm-eval queue {q.status()}", flush=True)
            procs = []
            for i, gpus in enumerate(replica_gpus(runtime, unit)):
                cmd = unit_commands(plan, unit, rows, shard=shard0 + i)[0]
                extra = {"CUDA_VISIBLE_DEVICES": gpus} if gpus is not None else {}
                t0 = time.time()
                proc, argv = start(runtime, plan, cmd, extra)
                procs.append((proc, argv, t0, f"[r{i}] " if unit.engine.replicas > 1 else ""))
            results = [None] * len(procs)
            threads = [threading.Thread(target=lambda j=j: results.__setitem__(
                j, wait(procs[j][0], runtime.hang_minutes, procs[j][3]))) for j in range(len(procs))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for (proc, argv, t0, _), (code, hung) in zip(procs, results):
                record["commands"].append({"argv": argv, "rc": code, "hung": hung, "seconds": round(time.time() - t0, 1)})
                rc = rc or (1 if code else 0)
            record["queue_after"] = q.status()
        else:
            for cmd in cmds:
                t0 = time.time()
                proc, argv = start(runtime, plan, cmd)
                code, hung = wait(proc, runtime.hang_minutes)
                d, m = mark_done(plan, cmd)
                done += d
                missing += m
                cache = cmd.env.get("OELLM_GEN_CACHE")
                if cache and not m:  # every row has its result: the call's generations are not needed again
                    for f in glob.glob(glob.escape(cache) + ".*"):
                        os.unlink(f)
                record["commands"].append({"argv": argv, "rc": code, "hung": hung, "seconds": round(time.time() - t0, 1),
                                           "rows_done": [r.name for r in d], "rows_missing": [r.name for r in m]})
                if code:
                    print(f"[worker] exit {code}" + (f"; {len(m)} row(s) without a result: "
                          + " ".join(r.name for r in m) if m else "; every row has its result"), flush=True)
            rc = 1 if missing and not missing_ok else 0
    finally:
        runtime.cleanup()
        runtime.close_tmp()
        record["end_utc"] = provenance.now()
        record["rows"] = {"done": [r.name for r in done], "missing": [r.name for r in missing]}
        record["rc"] = rc
        path = provenance.write_record(layout.provenance_dir, unit.id, record)
        print(f"[worker] {unit.id}: rc {rc}, provenance {path}", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--runtime", required=True)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--unit", action="append", help="unit id (repeatable)")
    sel.add_argument("--index", type=int, help="the unit at this position (a Slurm array task id)")
    sel.add_argument("--all", action="store_true", help="every unit, one after another")
    ap.add_argument("--print", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--no-probe", action="store_true", help="skip the harness version probe")
    ap.add_argument("--missing-ok", action="store_true",
                    help="exit 0 although rows are left without a result (callers that retry failed "
                         "jobs: the same call would fail again; the rows stay unmarked for a relaunch)")
    a = ap.parse_args(argv)
    try:
        plan, runtime = read_plan(a.plan), read_runtime(a.runtime)
        if a.all:
            units = plan.units
        elif a.index is not None:
            if not 0 <= a.index < len(plan.units):
                raise ValueError(f"--index {a.index}: the plan has {len(plan.units)} units")
            units = [plan.units[a.index]]
        else:
            units = [plan.unit(u) for u in a.unit]
    except (ValueError, KeyError, OSError) as e:
        print(f"[worker] {e}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_term)
    rc = 0
    for unit in units:
        rc = run_unit(plan, runtime, unit, plan_path=str(Path(a.plan).resolve()),
                      runtime_path=str(Path(a.runtime).resolve()), probe=not a.no_probe, dry=a.print,
                      missing_ok=a.missing_ok) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
