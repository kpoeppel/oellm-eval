"""Per-task raw results as one CSV, the input of summarize-evals (github.com/spyysalo/summarize-evals).

One row per (checkpoint, task, n_shot, metric, filter): every task and subtask a harness wrote, value
and stderr as written (0-1 for accuracies; BLEU and chrF 0-100). No primary-metric choice, no
aggregation beyond what the harness reports. Merged from the JUPITER collector
(/e/project1/e-sta-openeurollm/vanosch1/v1-vs-v2/anneal/scripts/collect_raw.py, 2026-11-07) and its
MN5 copy; same columns and rules:

  lm-eval    results_*.json / <hash>_<date>.json with ``configs``
  Evalchemy  results json without ``configs``
  lighteval  results_*.json with ``config_general``. flores' chrF++ and chrF carry filter "rescored" (the
             corrected corpus score): from the run's chrf_rescored.csv (lighteval_rescore_chrf.py) for
             unpatched lighteval, or lighteval's own with --chrf-patched (the patched image). BLEU is
             always lighteval's. ``<suite>:_average`` rows (a grouped call's mean) are dropped.

Inputs: ``--runs`` (oellm-eval run dirs: results/ below; the checkpoint from each file's model path)
and ``--root DIR=CHECKPOINT`` (a worker results root, layout.py: results/, lighteval/, evalchemy/;
every file there is that checkpoint's -- the lm-eval session runner passes lm-eval a model object,
so its files name no model). A result in more than one file (reruns) is taken from the newest;
n_files says how many existed. Dropped: sib200 acc_norm (always the longest label).

    python -m oellm.reports.collect --root <out>/<model>=<model> [--runs <dir> --checkpoints c ...] \\
        [--chrf-patched] [--datasets <csv>] -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

COLS = ["checkpoint", "task", "n_shot", "harness", "backend", "is_group", "metric", "filter", "value", "stderr",
        "n_samples", "results_file", "result_time", "n_files"]
# summarize-evals parses the language from the task name: lm-eval's group name has an underscore
CSV_SPELLING = {"include_base_44_north_macedonian": "include_base_44_north macedonian"}
LMEVAL_OBJECT_MODELS = (None, "NodeEPVLLM", "VLLM")


def lmeval_model(d):
    a = (d.get("config") or {}).get("model_args")
    p = a.get("pretrained") if isinstance(a, dict) else (re.search(r"pretrained=([^,]+)", str(a)) or [None, None])[1]
    return p.rstrip("/").split("/")[-1] if p else None


def lmeval_rows(d, f):
    lm = "configs" in d                       # lm-eval writes task configs; Evalchemy does not
    gs = d.get("group_subtasks", {})
    model = (d.get("config") or {}).get("model")
    for task, res in (d.get("results") or {}).items():
        shot = d.get("n-shot", {}).get(task)
        if shot is None:                      # groups record no n-shot: take a leaf's
            leaf = task
            while leaf in gs and gs[leaf]:
                leaf = gs[leaf][0]
            shot = d.get("configs", {}).get(leaf, {}).get("num_fewshot", 0) if lm else 0
        n = d.get("n-samples", {}).get(task, {}).get("effective", "") if lm else res.get("num_total", "")
        if lm and n == "" and task in gs:     # groups record no count: sum their leaves'
            leaves, todo = [], list(gs[task])
            while todo:
                t = todo.pop()
                todo.extend(gs[t]) if gs.get(t) else leaves.append(t)
            ns = [d.get("n-samples", {}).get(t, {}).get("effective") for t in leaves]
            n = sum(ns) if ns and None not in ns else ""
        for key, val in res.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            if lm:
                if "_stderr" in key or key in ("alias", "sample_len"):
                    continue
                metric, _, filt = key.partition(",")
                se = res.get(f"{metric}_stderr,{filt}") if filt else None
            else:
                if key.endswith("_std_err") or key in ("num_total", "num_repeat"):
                    continue
                metric, filt = key, ""
                se = res.get(key.replace("_avg", "_std_err")) if key.endswith("_avg") else None
            yield {"task": task, "n_shot": shot, "harness": "lm-eval" if lm else "Evalchemy",
                   "backend": "vllm" if model in LMEVAL_OBJECT_MODELS else model,
                   "is_group": "yes" if task in gs else "", "metric": metric, "filter": filt, "value": val,
                   "stderr": se if isinstance(se, (int, float)) else "", "n_samples": n,
                   "results_file": f, "result_time": d.get("date") or os.path.getmtime(f)}


def lighteval_backend(d) -> str:
    """vllm when the model config is vLLM's (it has a GPU memory share), else hf/accelerate."""
    mc = (d.get("config_general") or {}).get("model_config") or {}
    return "vllm" if "gpu_memory_utilization" in mc else "hf/accelerate"


def lighteval_rows(d, f, rescored, sizes, chrf_patched):
    when = os.path.getmtime(f)                # lighteval's start/end times are not wall-clock
    backend = lighteval_backend(d)
    for name, res in d["results"].items():
        if name == "all":
            continue
        task, shot = name.rsplit("|", 1)
        if task.endswith(":_average"):        # a grouped call's mean over its tasks
            continue
        n = rescored[name]["n"] if name in rescored else sizes.get((task, shot), "")
        base = {"task": task, "n_shot": int(shot), "harness": "lighteval", "backend": backend,
                "is_group": "", "results_file": f, "result_time": when, "n_samples": n}
        if task.startswith("flores200:"):
            if not (rescored or chrf_patched):
                raise SystemExit(f"{f}: flores results but no chrf_rescored.csv (lighteval_rescore_chrf.py); "
                                 "--chrf-patched if the run used the patched lighteval")
            for m in ("chrf++", "chrf"):
                if rescored:
                    val, se = float(rescored[name][m]), ""
                elif m in res:                    # patched lighteval: its own score is the corrected one
                    val, se = res[m], res.get(f"{m}_stderr", "")
                else:
                    continue
                yield {**base, "metric": m, "filter": "rescored", "value": val, "stderr": se}
            keep = {"bleu", "bleu_1", "bleu_4"}
        else:
            keep = {k for k in res if not k.endswith("_stderr")}
        for m in sorted(keep & set(res)):
            yield {**base, "metric": m, "filter": "none", "value": res[m], "stderr": res.get(f"{m}_stderr", "")}


def _files(patterns: list[str]) -> list[str]:
    out = set()
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if "_artifacts/" in p or Path(p).name.startswith("samples_") or "/details/" in p:
                continue
            out.add(str(Path(p).resolve()))
    return sorted(out)


def collect(*, runs: list[str] = (), checkpoints: list[str] = (), roots: dict[str, str] | None = None,
            chrf_patched: bool = False, datasets: str | None = None, lmeval_object: bool = False) -> list[dict]:
    sizes = {(r["task"], r["n_shot"]): r["n_samples"] for r in csv.DictReader(open(datasets))} if datasets else {}
    best, files = {}, {}
    sources = [(run, None) for run in runs] + [(root, ckpt) for root, ckpt in (roots or {}).items()]
    for run, fixed in sources:
        rp = Path(run) / "chrf_rescored.csv"
        rescored = {r["task"]: r for r in csv.DictReader(open(rp))} if rp.exists() else {}
        if fixed is None:  # an oellm-eval run dir: full split when its sbatch has LIMIT=""
            sb = Path(run) / "submit_evals.sbatch"
            full = bool(sb.exists() and re.search(r'^LIMIT=""$', sb.read_text(), re.M))
            paths = _files([f"{run}/results/**/*.json"])
        else:              # a worker results root: a limit run's root ends in _limit<l>
            full = not re.search(r"_limit[0-9.]+$", Path(run).name)
            paths = _files([f"{run}/results/**/*.json", f"{run}/evalchemy/**/results_*.json",
                            f"{run}/lighteval/**/results_*.json"])
        for f in paths:
            try:
                d = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "results" not in d:
                continue
            if "config_general" in d:
                m = fixed or os.path.basename(d["config_general"]["model_config"]["model_name"].rstrip("/"))
                if fixed is None and m not in checkpoints:
                    continue
                rows = lighteval_rows(d, f, rescored, sizes if full else {}, chrf_patched)
            elif (d.get("config") or {}).get("model") == "vllm" or (
                    (fixed or lmeval_object) and "configs" in d
                    and (d.get("config") or {}).get("model") in LMEVAL_OBJECT_MODELS):
                m = fixed or lmeval_model(d) or (checkpoints[0] if lmeval_object and len(checkpoints) == 1 else None)
                if fixed is None and m not in checkpoints:
                    continue
                rows = ({**r, "n_samples": r["n_samples"] if r["n_samples"] != "" or not full
                         else sizes.get((r["task"], str(r["n_shot"])), "")} for r in lmeval_rows(d, f))
            else:
                continue
            for row in rows:
                if row["task"].startswith("sib200_") and row["metric"] == "acc_norm":
                    continue
                for a, b in CSV_SPELLING.items():
                    row["task"] = row["task"].replace(a, b)
                k = (m, row["task"], str(row["n_shot"]), row["metric"], row["filter"])
                files.setdefault(k[:3], set()).add(f)
                if k not in best or float(row["result_time"]) >= float(best[k]["result_time"]):
                    best[k] = {"checkpoint": m, **row}
    rows = sorted(best.values(), key=lambda r: (r["checkpoint"], r["task"], int(r["n_shot"]), r["metric"], r["filter"]))
    return [{**r, "n_files": len(files[(r["checkpoint"], r["task"], str(r["n_shot"]))])} for r in rows]


def write_csv(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="oellm-eval run directories")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="model dir basenames (for --runs)")
    ap.add_argument("--root", action="append", default=[], metavar="DIR=CHECKPOINT",
                    help="a worker results root and its checkpoint name (repeatable)")
    ap.add_argument("--chrf-patched", action="store_true",
                    help="lighteval runs without chrf_rescored.csv are from the chrF-patched lighteval")
    ap.add_argument("--datasets", help="datasets CSV (task, n_shot, n_samples): sample counts")
    ap.add_argument("--lmeval-object", action="store_true",
                    help="--runs: lm-eval files without config.model belong to the one checkpoint")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.runs and not a.checkpoints:
        ap.error("--runs needs --checkpoints")
    roots = {}
    for spec in a.root:
        d, _, c = spec.rpartition("=")
        if not d or not c:
            ap.error(f"--root {spec!r}: DIR=CHECKPOINT")
        roots[d] = c
    rows = collect(runs=a.runs, checkpoints=a.checkpoints, roots=roots, chrf_patched=a.chrf_patched,
                   datasets=a.datasets, lmeval_object=a.lmeval_object)
    write_csv(rows, a.out)
    print(f"{a.out}: {len(rows)} rows, checkpoints {sorted({r['checkpoint'] for r in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
