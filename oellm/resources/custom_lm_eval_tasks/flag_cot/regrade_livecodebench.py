"""Grade livecodebench_cont generations offline: pass@1 / pass@4 as the task reports them.

    python3 regrade_livecodebench.py <file.jsonl> [--runner flag|evalchemy] [--workers N] [--out verdicts.jsonl]

<file.jsonl> is either the dump the task writes before grading (<run>/generations/
livecodebench_cont_<job>.jsonl: task_id, generations) or lm-eval's samples file
(samples_livecodebench_cont_*.jsonl: doc, resps). Runs in the eval image with the FLAG HF cache
(the problems and their private tests are loaded as in the task). --runner evalchemy grades with
Evalchemy's own runner (lcb_run: no per-test time limit, no memory cap) for comparison.
--out writes the per-sample verdicts.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def _utils():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")
    spec = importlib.util.spec_from_file_location("flag_cot_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--runner", choices=["flag", "evalchemy"], default="flag")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out")
    args = ap.parse_args()

    utils = _utils()
    generations = {}
    for line in open(args.file):
        row = json.loads(line)
        if "resps" in row:  # lm-eval samples file
            resps = row["resps"]
            generations[row["doc"]["task_id"]] = resps[0] if resps and isinstance(resps[0], list) else resps
        else:
            generations[row["task_id"]] = row["generations"]
    docs = {d["task_id"]: d for d in utils.load_livecodebench()["test"] if d["task_id"] in generations}
    missing = set(generations) - set(docs)
    if missing:
        sys.exit(f"{len(missing)} task_ids not in the dataset, e.g. {sorted(missing)[:3]}")

    def grade(job):
        doc, text = job
        if args.runner == "flag":
            return utils._lcb_grade(doc, text)
        lcb = utils._livecodebench()
        problem = dict(doc, test=utils._lcb_tests(doc["task_id"]))
        try:
            res = lcb.lcb_run(problem, utils._lcb_program(doc, text), utils.LCB_TEST_TIMEOUT, not doc["is_stdin"])
        except lcb.LCBInfrastructureError:
            return None
        return float(bool(res) and all(r[0] for r in res))

    jobs = [(docs[t], text) for t, samples in generations.items() for text in samples]
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(grade, jobs))
    elapsed = time.time() - start

    per_problem, i = {}, 0
    for t, samples in generations.items():
        per_problem[t] = verdicts[i : i + len(samples)]
        i += len(samples)
    metrics = [utils._pass_metrics([v or 0.0 for v in vs], [4]) for vs in per_problem.values()]
    lost = sum(v is None for v in verdicts)
    print(json.dumps({
        "runner": args.runner,
        "problems": len(per_problem),
        "samples": len(jobs),
        "seconds": round(elapsed),
        "without_verdicts": lost,
        "pass@1": sum(m["pass@1"] for m in metrics) / len(metrics),
        "pass@4": sum(m["pass@4"] for m in metrics) / len(metrics),
    }))
    if args.out:
        with open(args.out, "w") as fh:
            for t, vs in per_problem.items():
                fh.write(json.dumps(dict(task_id=t, verdicts=vs)) + "\n")


if __name__ == "__main__":
    main()
