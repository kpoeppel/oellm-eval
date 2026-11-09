"""oellm.reports: the collector (worker roots and oellm-eval run dirs), the missing-row check and the
launch manifest."""

from __future__ import annotations

import csv
import json
import os
import time

from oellm.reports import check, collect, manifest
from oellm.runner.layout import Row
from oellm.runner.plan import GroupSettings, SessionSettings, build_plan
from oellm.runner.runtime import Runtime

LM = {"results": {"mmlu": {"alias": "mmlu", "acc,none": 0.5, "acc_stderr,none": 0.01},
                  "mmlu_x": {"acc,none": 0.4, "acc_stderr,none": 0.02},
                  "include_base_44_north_macedonian": {"acc,none": 0.3}},
      "group_subtasks": {"mmlu": ["mmlu_x"]}, "n-shot": {"mmlu_x": 5, "include_base_44_north_macedonian": 0},
      "configs": {"mmlu_x": {"num_fewshot": 5}}, "n-samples": {"mmlu_x": {"effective": 100}},
      "config": {"model": None, "model_args": None}, "date": 100.0}
EC = {"results": {"AIME24": {"accuracy_avg": 0.2, "accuracy_std_err": 0.05, "num_total": 30, "num_repeat": 10}},
      "config": {"model": "vllm", "model_args": "pretrained=/v/m,dtype=bfloat16"}}
LE = {"results": {"flores200:deu_Latn-eng_Latn|4": {"chrf++": 60.0, "bleu": 30.0, "bleu_1": 0.6},
                  "xcopa:it|0": {"acc": 0.7, "acc_stderr": 0.02}, "flores200:_average|4": {"chrf++": 60.0},
                  "all": {"acc": 0.7}},
      "config_general": {"model_config": {"model_name": "/x/m", "gpu_memory_utilization": 0.9}}}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _root(tmp_path):
    root = tmp_path / "out" / "m"
    _write(root / "results" / "mmlu_5" / "x" / "results_1.json", LM)
    _write(root / "results" / "mmlu_5" / "x" / "samples_mmlu_x.json", {"results": {"bogus": {"acc": 1}}})
    _write(root / "evalchemy" / "evalchemy_AIME24_0" / "m" / "results_2.json", EC)
    _write(root / "lighteval" / "lighteval_group_x" / "results" / "m" / "results_3.json", LE)
    return root


def test_collect_a_worker_root(tmp_path):
    root = _root(tmp_path)
    rows = collect.collect(roots={str(root): "m"}, chrf_patched=True)
    by = {(r["task"], r["metric"], r["filter"]): r for r in rows}
    assert {r["checkpoint"] for r in rows} == {"m"}
    assert by[("mmlu", "acc", "none")]["is_group"] == "yes" and by[("mmlu", "acc", "none")]["n_shot"] == 5
    assert by[("mmlu", "acc", "none")]["n_samples"] == 100 and by[("mmlu", "acc", "none")]["backend"] == "vllm"
    assert ("include_base_44_north macedonian", "acc", "none") in by, "summarize-evals' spelling"
    assert by[("AIME24", "accuracy_avg", "")]["stderr"] == 0.05 and by[("AIME24", "accuracy_avg", "")]["harness"] == "Evalchemy"
    fl = by[("flores200:deu_Latn-eng_Latn", "chrf++", "rescored")]
    assert fl["value"] == 60.0 and fl["backend"] == "vllm" and fl["n_shot"] == 4
    assert ("flores200:_average", "chrf++", "rescored") not in by and not any(t == "bogus" for t, _, _ in by)
    assert by[("xcopa:it", "acc", "none")]["stderr"] == 0.02


def test_the_newest_rerun_wins_and_n_files_counts(tmp_path):
    root = _root(tmp_path)
    newer = dict(EC, results={"AIME24": {"accuracy_avg": 0.3}})
    p = root / "evalchemy" / "evalchemy_AIME24_0" / "m" / "results_9.json"
    _write(p, newer)
    os.utime(p, (time.time() + 60, time.time() + 60))
    rows = collect.collect(roots={str(root): "m"}, chrf_patched=True)
    aime = next(r for r in rows if r["task"] == "AIME24" and r["metric"] == "accuracy_avg")
    assert aime["value"] == 0.3 and aime["n_files"] == 2


def test_flores_without_a_chrf_fix_is_an_error(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="chrf"):
        collect.collect(roots={str(_root(tmp_path)): "m"})


def test_oellm_eval_run_dirs_by_model_name(tmp_path):
    run = tmp_path / "run"
    _write(run / "results" / "a.json", EC)
    _write(run / "results" / "b.json", dict(EC, config={"model": "vllm", "model_args": "pretrained=/v/other"}))
    rows = collect.collect(runs=[str(run)], checkpoints=["m"])
    assert {r["checkpoint"] for r in rows} == {"m"} and len(rows) == 1


def test_check_reports_planned_rows_without_a_result(tmp_path):
    root = _root(tmp_path)
    out = tmp_path / "x.csv"
    collect.write_csv(collect.collect(roots={str(root): "m"}, chrf_patched=True), out)
    rows = [Row("vllm/mmlu/5", "mmlu", 5, "lm-eval-harness"), Row("vllm/arc/25", "arc", 25, "lm-eval-harness"),
            Row("vllm/include_base_44_north_macedonian/0", "include_base_44_north_macedonian", 0, "lm-eval-harness"),
            Row("lighteval/xcopa:it/0", "xcopa:it", 0, "lighteval")]
    plan = build_plan(model_name="m", model_dir="/x/m", view_dir=None, out=str(tmp_path / "plans"), rows=rows,
                      lm_eval=SessionSettings(), lighteval=GroupSettings())
    pp = plan.write(tmp_path / "plan.json")
    miss = tmp_path / "missing.csv"
    assert check.main(["--csv", str(out), "--plan", str(pp), "-o", str(miss)]) == 0
    assert [r["name"] for r in csv.DictReader(miss.open())] == ["vllm/arc/25"]
    assert check.main(["--csv", str(out), "--plan", str(pp), "--strict"]) == 1


def test_launch_manifest(tmp_path):
    rows = [Row("vllm/mmlu/5", "mmlu", 5, "lm-eval-harness"), Row("vllm/AIME24/0", "AIME24", 0, "evalchemy")]
    plan = build_plan(model_name="m", model_dir="/x/m", view_dir="/v/m", out=str(tmp_path / "out"), rows=rows,
                      lm_eval=SessionSettings(), evalchemy=GroupSettings())
    d = tmp_path / "launch"
    pp = plan.write(d / "plan.json")
    patched = tmp_path / "patched.py"
    patched.write_text("x")
    rp = Runtime(hash_files=[str(patched)]).write(d / "runtime.json")
    m = json.loads(manifest.write_manifest(d, pp, rp, caller={"sync_stamp": ["a b"]}).read_text())
    assert m["plan"]["sha256"] and m["plan"]["rows"] == 2 and m["plan"]["lm_eval_items"] == 1
    assert {u["id"]: u["model"] for u in m["units"]} == {"lm0": "/x/m", "ec1": "/v/m"}
    assert len(m["runtime"]["hash_files"][str(patched)]) == 64 and m["code"]["caller"] == {"sync_stamp": ["a b"]}
    assert set(m["files"]) == {"plan.json", "runtime.json"}
