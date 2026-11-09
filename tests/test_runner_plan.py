"""oellm.runner.plan / commands / runtime: planning, the plan file, the harness commands and how a
runtime starts them."""

from __future__ import annotations

import json
import os
import sys

import pytest

from oellm.runner import commands
from oellm.runner.layout import Layout, Row, group_key, results_root
from oellm.runner.plan import Engine, GroupSettings, Plan, SessionSettings, build_plan, rows_from_task_groups
from oellm.runner.runtime import OELLM_ROOT, Runtime

ROWS = [Row("vllm/arc_challenge/25", "arc_challenge", 25, "lm-eval-harness"),
        Row("vllm/mmlu/5", "mmlu", 5, "lm-eval-harness"),
        Row("vllm/AIME24/0", "AIME24", 0, "evalchemy"),
        Row("vllm/HumanEval/0", "HumanEval", 0, "evalchemy"),
        Row("lighteval/flores200:deu_Latn-eng_Latn/4", "flores200:deu_Latn-eng_Latn", 4, "lighteval"),
        Row("lighteval/xcopa:it/0", "xcopa:it", 0, "lighteval")]


def _plan(tmp_path, **kw) -> Plan:
    args = dict(model_name="m", model_dir=str(tmp_path / "m"), view_dir=str(tmp_path / "view"), out=str(tmp_path / "out"),
                rows=ROWS, lm_eval=SessionSettings(workers=2), lighteval=GroupSettings(groups=1),
                evalchemy=GroupSettings(groups=2), engine_times={"vllm/mmlu/5": 1000.0, "vllm/arc_challenge/25": 60.0})
    args.update(kw)
    return build_plan(**args)


# ---------------------------------------------------------------- rows and planning


def test_flag_evals_rows_come_from_the_task_groups():
    rows = rows_from_task_groups({"vllm": "flag-evals-vllm", "lighteval": "flag-evals-lighteval"})
    assert len(rows) == 438 and len({r.name for r in rows}) == 438
    by = {r.harness: 0 for r in rows}
    for r in rows:
        by[r.harness] += 1
        assert Row.parse_name(r.name) == (r.name.split("/", 1)[0], r.task, r.n_shot)
    assert by["lighteval"] == 73 and by["evalchemy"] == 8
    assert "vllm/include_base_44_north_macedonian/0" in {r.name for r in rows}


def test_plan_units_items_and_roundtrip(tmp_path):
    plan = _plan(tmp_path)
    kinds = [(u.id, u.kind, len(u.rows)) for u in plan.units]
    assert kinds == [("lm0", "lm_eval_session", 0), ("lm1", "lm_eval_session", 0), ("le1", "lighteval", 2),
                     ("ec1", "evalchemy", 1), ("ec2", "evalchemy", 1)]
    assert [i for i, _ in plan.lm_items][:4] == [f"vllm/mmlu/5@{k}of4" for k in range(4)]
    assert plan.root == str(results_root(tmp_path / "out", "m"))
    again = Plan.from_json(plan.to_json())
    assert again == plan
    data = json.loads(plan.to_json())
    data["units"][0]["engine"]["tpp"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        Plan.from_json(json.dumps(data))


def test_done_rows_are_left_out_and_limit_has_its_own_root(tmp_path):
    root = results_root(tmp_path / "out", "m")
    marker = Layout(root).marker(ROWS[4])
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    plan = _plan(tmp_path)
    le = next(u for u in plan.units if u.kind == "lighteval")
    assert [r.name for r in le.rows] == ["lighteval/xcopa:it/0"]
    assert len(plan.rows) == len(ROWS), "the plan keeps all rows; units hold the ones to do"
    lim = _plan(tmp_path, limit=20)
    assert lim.root.endswith("m_limit20") and all("@" not in i for i, _ in lim.lm_items), "no parts in a test run"


def test_settings_are_checked(tmp_path):
    with pytest.raises(ValueError, match="expert_parallel"):
        _plan(tmp_path, lm_eval=SessionSettings(engine=Engine(dp=4, expert_parallel=False)))
    with pytest.raises(ValueError, match="accelerate"):
        _plan(tmp_path, evalchemy=GroupSettings(backend="accelerate"))
    plan = _plan(tmp_path, lm_eval=None, evalchemy=None)
    assert [u.kind for u in plan.units] == ["lighteval"] and plan.lm_items == []
    few = _plan(tmp_path, lm_eval=SessionSettings(workers=50, split_s=0))
    assert sum(u.kind == "lm_eval_session" for u in few.units) == 2, "no more sessions than items"


def test_group_key_matches_the_old_runners():
    rows = ROWS[4:]
    assert group_key(rows).startswith("group_") and len(group_key(rows)) == 16
    assert group_key(rows[:1]) == "flores200_deu_Latn-eng_Latn_4"


# ---------------------------------------------------------------- commands


def test_lighteval_loads_the_plain_model_saves_details_and_sets_ep(tmp_path):
    plan = _plan(tmp_path)
    le = next(u for u in plan.units if u.kind == "lighteval")
    (c,) = commands.unit_commands(plan, le, le.rows)
    assert c.argv[:4] == ["python", "-m", "oellm.runner.lighteval_cli", "vllm"]
    assert c.argv[4].startswith(f"model_name={plan.model_dir},") and plan.view_dir not in c.argv[4]
    assert c.argv[5] == "flores200:deu_Latn-eng_Latn|4,xcopa:it|0"
    assert "--save-details" in c.argv and c.env["OELLM_EP"] == "1"
    assert c.argv[c.argv.index("--output-dir") + 1] == str(plan.layout.lighteval_out(le.rows))
    plan2 = _plan(tmp_path, lighteval=GroupSettings(backend="accelerate"))
    le2 = next(u for u in plan2.units if u.kind == "lighteval")
    (c2,) = commands.unit_commands(plan2, le2, le2.rows)
    assert c2.argv[3:5] == ["accelerate", f"model_name={plan.model_dir},trust_remote_code=True,batch_size=32"]
    assert "OELLM_EP" not in c2.env


def test_evalchemy_one_call_per_task_on_the_view(tmp_path):
    plan = _plan(tmp_path, evalchemy=GroupSettings(groups=1, env={"OELLM_VLLM_ENFORCE_EAGER": "0"},
                                                   engine=Engine(max_model_length=40960)))
    ec = next(u for u in plan.units if u.kind == "evalchemy")
    cs = commands.unit_commands(plan, ec, ec.rows)
    assert [c.argv[c.argv.index("--tasks") + 1] for c in cs] == ["AIME24", "HumanEval"]
    ma = cs[0].argv[cs[0].argv.index("--model_args") + 1]
    assert ma.startswith(f"pretrained={plan.view_dir},") and "enforce_eager=False" in ma
    assert ma.endswith(",enable_expert_parallel=True,max_model_len=40960")
    assert cs[0].cwd == "evalchemy" and cs[0].env == {"OELLM_VLLM_ENFORCE_EAGER": "0"}


def test_lm_eval_session_command(tmp_path):
    plan = _plan(tmp_path, limit=20)
    lm = plan.units[0]
    (c,) = commands.unit_commands(plan, lm, [], shard=7)
    a = c.argv
    assert a[:3] == ["python", "-m", "oellm.runner.lm_eval_queue"]
    assert a[a.index("--model") + 1] == plan.model_dir and a[a.index("--root") + 1] == plan.root
    assert a[a.index("--queue") + 1] == str(plan.layout.queue_dir) and a[a.index("--shard") + 1] == "7"
    assert (a[a.index("--tp") + 1], a[a.index("--dp") + 1], a[a.index("--ep") + 1]) == ("1", "4", "1")
    assert a[a.index("--request-batch-size") + 1] == "256" and a[a.index("--limit") + 1] == "20"


# ---------------------------------------------------------------- runtime


def test_local_runtime_puts_this_checkout_first(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    rt = Runtime(python="/py", env={"A": "1"}, harness_env={"lighteval": {"B": "2"}}, pythonpath=["/stage"],
                 harness_pythonpath={"lighteval": ["/pylibs_le"]}, dirs={"evalchemy": "/ec"})
    argv, env, cwd = rt.wrap(["python", "-m", "x"], "lighteval", {"C": "3"}, None, [])
    assert argv == ["/py", "-m", "x"] and cwd is None
    assert env["PYTHONPATH"] == f"{OELLM_ROOT}:/pylibs_le:/stage:/inherited"
    assert (env["A"], env["B"], env["C"]) == ("1", "2", "3")
    _, env2, cwd2 = rt.wrap(["python", "-m", "eval.eval"], "evalchemy", {}, "evalchemy", [])
    assert cwd2 == "/ec" and "B" not in env2 and env2["PYTHONPATH"] == f"{OELLM_ROOT}:/stage:/inherited"
    with pytest.raises(ValueError, match="no directory"):
        Runtime().wrap(["python"], "evalchemy", {}, "evalchemy", [])
    monkeypatch.setenv("INFER_PY", "/node/venv/python")
    assert Runtime(python="${INFER_PY}").wrap(["python", "-V"], "lighteval", {}, None, [])[0] == ["/node/venv/python", "-V"]


def test_apptainer_runtime_binds_and_passes_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    rt = Runtime(kind="apptainer", python="python", images={"lm-eval-harness": "/v.sif", "evalchemy": "/v.sif",
                                                            "lighteval": "/l.sif"},
                 apptainer_args=["--nv", "--cleanenv"], binds=["/patch.py:/opt/x.py:ro"],
                 dirs={"evalchemy": "/opt/evalchemy"}, env={"HF_HOME": "/hf,with,commas"})
    rt.check()
    argv, env, cwd = rt.wrap(["python", "-m", "eval.eval"], "evalchemy", {"K": "v"}, "evalchemy", ["/model", "/model"])
    assert argv[:4] == ["apptainer", "exec", "--nv", "--cleanenv"] and cwd is None
    binds = argv[argv.index("--bind") + 1].split(",")
    assert binds == ["/model:/model", f"{OELLM_ROOT}:{OELLM_ROOT}", "/patch.py:/opt/x.py:ro"]
    assert argv[argv.index("--pwd") + 1] == "/opt/evalchemy"
    i = argv.index("/v.sif")
    assert argv[i + 1] == "env" and argv[-3:] == ["python", "-m", "eval.eval"]
    inner = dict(kv.split("=", 1) for kv in argv[i + 2:-3])
    assert inner["HF_HOME"] == "/hf,with,commas" and inner["K"] == "v" and inner["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert inner["PYTHONPATH"].split(":")[0] == OELLM_ROOT
    with pytest.raises(ValueError, match="no image"):
        Runtime(kind="apptainer", images={"lighteval": "/l.sif"}).check()


def test_runtime_json_is_strict(tmp_path):
    rt = Runtime(name="x", pythonpath=["/a"])
    assert Runtime.from_json(rt.to_json()) == rt
    with pytest.raises(ValueError, match="unknown keys"):
        Runtime.from_json(json.dumps({"kind": "local", "pyhton": "x"}))


def test_cleanup_and_prune(tmp_path, monkeypatch):
    (tmp_path / "shm").mkdir()
    (tmp_path / "shm" / "oellm_engram_1").write_text("x")
    (tmp_path / "dest" / "typer").mkdir(parents=True)
    monkeypatch.setenv("DEST", str(tmp_path / "dest"))
    rt = Runtime(cleanup_globs=[str(tmp_path / "shm" / "oellm_engram_*")], prune={"lighteval": ["$DEST/typer"]})
    assert rt.cleanup() == [str(tmp_path / "shm" / "oellm_engram_1")]
    assert rt.prune_for("evalchemy") == [] and rt.prune_for("lighteval") == [str(tmp_path / "dest" / "typer")]
    assert not (tmp_path / "dest" / "typer").exists()


def test_runner_modules_import_only_the_standard_library():
    """A worker runs on a bare host python3 (JUPITER: it starts the images itself)."""
    import subprocess

    code = ("import sys, oellm.runner.worker, oellm.runner.cli\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('yaml', 'pandas', 'jsonargparse', 'datasets'))\n"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": OELLM_ROOT},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_dp_without_ep_is_for_vllms_own_data_parallelism(tmp_path):
    plan = _plan(tmp_path, evalchemy=GroupSettings(engine=Engine(tp=1, dp=4, expert_parallel=False)))
    ec = next(u for u in plan.units if u.kind == "evalchemy")
    (c, _) = commands.unit_commands(plan, ec, ec.rows)
    ma = c.argv[c.argv.index("--model_args") + 1]
    assert ",data_parallel_size=4,data_parallel_backend=mp" in ma and "expert_parallel" not in ma
    with pytest.raises(ValueError, match="replicas"):
        _plan(tmp_path, evalchemy=GroupSettings(engine=Engine(replicas=2)))


def test_contained_images_get_a_private_tmp_and_env_values_expand(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    rt = Runtime(kind="apptainer", images={h: "/v.sif" for h in ("lm-eval-harness", "lighteval", "evalchemy")},
                 apptainer_args=["--containall"], job_tmp_root=str(tmp_path / "jt"),
                 env={"HF_METRICS_CACHE": "/hf/metrics/job_${SLURM_JOB_ID}"})
    tmp = rt.open_tmp()
    assert tmp.startswith(str(tmp_path / "jt" / "oellm-eval-77.")) and os.path.isdir(f"{tmp}/home")
    argv, _, _ = rt.wrap(["python", "-c", "x"], "lighteval", {}, None, [])
    binds = argv[argv.index("--bind") + 1].split(",")
    assert f"{tmp}:/tmp" in binds and f"{tmp}:/var/tmp" in binds
    assert argv[argv.index("--home") + 1] == f"{tmp}/home:/tmp/home"
    assert "HF_METRICS_CACHE=/hf/metrics/job_77" in argv and "TMPDIR=/tmp" in argv
    assert "active_tmp" not in json.loads(rt.to_json())
    rt.close_tmp()
    assert not os.path.exists(tmp) and "--home" not in rt.wrap(["python"], "lighteval", {}, None, [])[0]


def test_runtime_cli_writes_a_checked_runtime(tmp_path):
    from oellm.runner import cli

    out = tmp_path / "rt.json"
    assert cli.main(["runtime", "--kind", "apptainer", "--image", "lm-eval-harness=/v.sif", "--image", "evalchemy=/v.sif",
                     "--image", "lighteval=/l.sif", "--args", "--nv --cleanenv --env HF_HUB_CACHE=/hf/hub",
                     "--harness-bind", "evalchemy=/p.py:/opt/x.py:ro", "--env", "HF_HOME=/hf", "--dir",
                     "evalchemy=/opt/evalchemy", "--job-tmp-root", "/tmp", "-o", str(out)]) == 0
    rt = Runtime.from_json(out.read_text())
    assert rt.apptainer_args == ["--nv", "--cleanenv", "--env", "HF_HUB_CACHE=/hf/hub"]
    assert rt.harness_binds == {"evalchemy": ["/p.py:/opt/x.py:ro"]} and rt.env == {"HF_HOME": "/hf"}
    with pytest.raises(ValueError, match="no image"):
        cli.main(["runtime", "--kind", "apptainer", "-o", str(out)])
