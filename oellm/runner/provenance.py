"""What ran, where and with which code: one JSON record per unit run (written by the worker at run
time, so it holds what actually executed, not what was planned).

    plan / runtime     their paths and sha256, the unit as planned, the runtime as used
    code               this oellm-eval checkout (git commit + dirty), $OELLM_CODE_STAMP (the
                       caller's code identity, e.g. oellm-autoexp's SYNC_STAMP lines)
    harnesses          package versions inside each runtime the unit used (a probe command)
    hash_files         sha256 of the runtime's listed files (patched harness files)
    machine            host, Slurm ids, CUDA_VISIBLE_DEVICES, ``nvidia-smi -L``
    commands           every command as started, its exit status and seconds
    rows               done / without a result after the unit
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from oellm.runner.runtime import OELLM_ROOT, Runtime

SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_STEP_ID", "SLURM_JOB_NODELIST",
              "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT")
DISTS = ("lm_eval", "lighteval", "vllm", "torch", "transformers", "datasets", "accelerate", "sacrebleu")
PROBE = ("import importlib.metadata as m, json, sys\n"
         "out = {}\n"
         "for d in sys.argv[1:]:\n"
         "    try:\n"
         "        out[d] = m.version(d)\n"
         "    except m.PackageNotFoundError:\n"
         "        pass\n"
         "print('OELLM_PROBE ' + json.dumps(out))\n")


def sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", OELLM_ROOT, *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def code_identity() -> dict:
    stamp = os.environ.get("OELLM_CODE_STAMP", "")
    return {"oellm_root": OELLM_ROOT, "oellm_commit": _git("rev-parse", "HEAD"),
            "oellm_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "stamp": stamp.splitlines() if stamp else []}


def machine() -> dict:
    gpus = ""
    try:
        gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"host": socket.gethostname(), "pid": os.getpid(),
            "slurm": {k: os.environ[k] for k in SLURM_KEYS if k in os.environ},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": gpus.splitlines()}


def probe_versions(runtime: Runtime, harness: str) -> dict:
    """Package versions as the harness's runtime sees them (one short command)."""
    argv, env, cwd = runtime.wrap(["python", "-c", PROBE, *DISTS], harness, {}, None, [])
    try:
        r = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    for line in r.stdout.splitlines():
        if line.startswith("OELLM_PROBE "):
            return json.loads(line[len("OELLM_PROBE "):])
    return {"error": f"exit {r.returncode}: {r.stderr.strip()[-400:]}"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_record(directory: str | Path, unit_id: str, record: dict) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    stamp = record.get("start_utc", now()).replace(":", "").replace("+0000", "Z")
    path = d / f"{unit_id}__{socket.gethostname()}__{stamp}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str) + "\n")
    tmp.replace(path)
    return path
