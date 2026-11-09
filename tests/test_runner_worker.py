"""oellm.runner.worker end to end: real worker processes on a local runtime whose lighteval and
evalchemy are small fakes (they write result files like the real ones), plus the lm-eval session
path with a stand-in command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from oellm.runner import commands, worker
from oellm.runner.layout import Row
from oellm.runner.plan import Engine, GroupSettings, SessionSettings, build_plan, init_queue
from oellm.runner.runtime import OELLM_ROOT, Runtime

FAKE_LIGHTEVAL = '''
import json, os, sys, time
from pathlib import Path

def vllm(model_args, tasks, output_dir=None, save_details=False, load_tasks_multilingual=False, max_samples=None,
         **kw):
    out = Path(output_dir)
    (out / "calls.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(out / "calls.jsonl", "a") as fh:
        fh.write(json.dumps({"model_args": model_args, "tasks": tasks, "save_details": save_details,
                             "ep": os.environ.get("OELLM_EP"), "pythonpath": os.environ["PYTHONPATH"]}) + "\\n")
    fail = os.environ.get("FAKE_FAIL", "")
    res = {t: {"acc": 1.0} for t in tasks.split(",") if not (fail and t.startswith(fail))}
    d = out / "results" / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_1.json").write_text(json.dumps({"results": res, "config_general": {"model_config": {}}}))
    if os.environ.get("FAKE_HANG"):
        print("[shutdown] engine down", flush=True)
        time.sleep(600)
    if fail:
        sys.exit(1)

accelerate = vllm
'''

FAKE_EVALCHEMY = '''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--model"); p.add_argument("--tasks"); p.add_argument("--model_args"); p.add_argument("--batch_size")
p.add_argument("--output_path"); p.add_argument("--limit")
a = p.parse_args()
d = Path(a.output_path) / "model"
d.mkdir(parents=True, exist_ok=True)
(d / "results_1.json").write_text(json.dumps({"results": {a.tasks: {"accuracy_avg": 0.5}},
                                              "config": {"model_args": a.model_args}}))
'''


@pytest.fixture
def fakes(tmp_path):
    root = tmp_path / "fakes"
    for rel, text in {"typer/__init__.py": "from . import models\n", "typer/models.py": "class OptionInfo:\n    pass\n",
                      "lighteval/__init__.py": "", "lighteval/main_vllm.py": FAKE_LIGHTEVAL,
                      "lighteval/main_accelerate.py": "from lighteval.main_vllm import accelerate\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(textwrap.dedent(text))
    ec = tmp_path / "evalchemy"
    (ec / "eval").mkdir(parents=True)
    (ec / "eval" / "__init__.py").write_text("")
    (ec / "eval" / "eval.py").write_text(FAKE_EVALCHEMY)
    rt = Runtime(python=sys.executable, pythonpath=[str(root)], dirs={"evalchemy": str(ec)}, hang_minutes=0.03,
                 hash_files=[str(ec / "eval" / "eval.py")])
    path = rt.write(tmp_path / "runtime.json")
    return rt, path


ROWS = [Row("lighteval/flores200:deu_Latn-eng_Latn/4", "flores200:deu_Latn-eng_Latn", 4, "lighteval"),
        Row("lighteval/xcopa:it/0", "xcopa:it", 0, "lighteval"),
        Row("vllm/AIME24/0", "AIME24", 0, "evalchemy"), Row("vllm/GPQADiamond/0", "GPQADiamond", 0, "evalchemy")]


def _plan(tmp_path, env=None, **kw):
    plan = build_plan(model_name="m", model_dir=str(tmp_path / "m"), view_dir=str(tmp_path / "view"),
                      out=str(tmp_path / "out"), rows=ROWS, lighteval=GroupSettings(env=env or {}),
                      evalchemy=GroupSettings(), **kw)
    return plan, plan.write(tmp_path / "plan.json")


def _worker(plan_path, rt_path, *args):
    env = {**os.environ, "PYTHONPATH": OELLM_ROOT, "OELLM_CODE_STAMP": "caller abc123 dirty=0"}
    return subprocess.run([sys.executable, "-m", "oellm.runner.worker", "--plan", str(plan_path), "--runtime",
                           str(rt_path), *args], env=env, capture_output=True, text=True, timeout=300)


def test_units_run_mark_rows_and_record_provenance(tmp_path, fakes):
    rt, rt_path = fakes
    plan, path = _plan(tmp_path)
    r = _worker(path, rt_path, "--all")
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(plan.layout.is_done(row) for row in ROWS)
    call = json.loads((plan.layout.lighteval_out(ROWS[:2]) / "calls.jsonl").read_text())
    assert call["model_args"].startswith(f"model_name={plan.model_dir},") and call["save_details"] is True
    assert call["ep"] == "1" and call["pythonpath"].split(":")[:2] == [OELLM_ROOT, rt.pythonpath[0]]
    ec_args = json.loads(next(plan.layout.evalchemy_out(ROWS[2]).rglob("results_*.json")).read_text())
    assert ec_args["config"]["model_args"].startswith(f"pretrained={plan.view_dir},")
    recs = sorted(plan.layout.provenance_dir.glob("*.json"))
    assert {p.name.split("__")[0] for p in recs} == {"le1", "ec1"}
    rec = json.loads(next(p for p in recs if p.name.startswith("le1")).read_text())
    assert rec["rc"] == 0 and rec["rows"]["done"] == [r.name for r in ROWS[:2]]
    assert rec["code"]["stamp"] == ["caller abc123 dirty=0"] and rec["machine"]["host"]
    assert len(rec["hash_files"][rt.hash_files[0]]) == 64
    assert isinstance(rec["harnesses"]["lighteval"], dict) and "error" not in rec["harnesses"]["lighteval"]
    assert rec["commands"][0]["argv"][0] == sys.executable
    again = _worker(path, rt_path, "--all")
    assert again.returncode == 0 and "every row is done" in again.stdout


def test_a_failed_call_leaves_its_rows_unmarked(tmp_path, fakes):
    _, rt_path = fakes
    plan, path = _plan(tmp_path, env={"FAKE_FAIL": "xcopa"})
    r = _worker(path, rt_path, "--unit", "le1", "--no-probe")
    assert r.returncode == 1 and "1 row(s) without a result: lighteval/xcopa:it/0" in r.stdout
    assert plan.layout.is_done(ROWS[0]) and not plan.layout.is_done(ROWS[1])
    assert _worker(path, rt_path, "--unit", "le1", "--no-probe", "--missing-ok").returncode == 0


def test_the_hang_watchdog_ends_a_command_silent_after_shutdown(tmp_path, fakes):
    _, rt_path = fakes
    plan, path = _plan(tmp_path, env={"FAKE_HANG": "1"})
    r = _worker(path, rt_path, "--index", "0", "--no-probe")
    assert "after vLLM's shutdown" in r.stdout
    rec = json.loads(next(plan.layout.provenance_dir.glob("le1__*.json")).read_text())
    assert rec["commands"][0]["hung"] is True and rec["commands"][0]["rc"] == 124
    assert plan.layout.is_done(ROWS[0]), "results written before the hang still count"


def test_print_and_bad_selections(tmp_path, fakes):
    _, rt_path = fakes
    _, path = _plan(tmp_path)
    r = _worker(path, rt_path, "--unit", "ec1", "--print")
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    assert [x["argv"][x["argv"].index("--tasks") + 1] for x in lines] == ["AIME24", "GPQADiamond"]
    assert _worker(path, rt_path, "--index", "9").returncode == 2
    assert _worker(path, rt_path, "--unit", "nope").returncode == 2


def test_lm_eval_session_replicas_share_the_queue(tmp_path, monkeypatch):
    rows = [Row(f"vllm/t{i}/0", f"t{i}", 0, "lm-eval-harness") for i in range(6)]
    plan = build_plan(model_name="m", model_dir=str(tmp_path / "m"), view_dir=None, out=str(tmp_path / "out"),
                      rows=rows, lm_eval=SessionSettings(workers=1, split_s=0,
                                                         engine=Engine(tp=1, dp=1, replicas=2, max_num_seqs=8)))
    assert init_queue(plan)["added"] == 6
    # a stand-in for lm_eval_queue: claims items until the queue is empty and writes their markers
    script = textwrap.dedent(f'''
        import json, os, sys
        sys.path.insert(0, {OELLM_ROOT!r})
        from oellm.runner.queue import Queue
        q = Queue({str(plan.layout.queue_dir)!r})
        done = {str(plan.layout.root / "done")!r}
        os.makedirs(done, exist_ok=True)
        while (c := q.claim()) is not None:
            task = c[1]["item"].split("/")[1]
            open(os.path.join(done, task + "_0.ok"), "w").write(os.environ["CUDA_VISIBLE_DEVICES"])
            q.complete(c[0])
    ''')
    real = commands.unit_commands
    monkeypatch.setattr(worker, "unit_commands", lambda p, u, r, shard=0: [
        commands.Command(["python", "-c", script], "lm-eval-harness")] if u.kind == "lm_eval_session" else real(p, u, r, shard))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rt = Runtime(python=sys.executable, gpus=2)
    assert worker.replica_gpus(rt, plan.units[0]) == ["0", "1"]
    assert worker.run_unit(plan, rt, plan.units[0], probe=False) == 0
    assert all(plan.layout.is_done(r) for r in rows)
    assert {(plan.layout.root / "done" / f"t{i}_0.ok").read_text() for i in range(6)} <= {"0", "1"}
    with pytest.raises(ValueError, match="replicas"):
        worker.replica_gpus(Runtime(gpus=1), plan.units[0])
