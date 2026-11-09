"""oellm.runner.gen_cache: finished generations survive a kill and a rerun generates only the rest
(a fake vLLM whose engine finishes one request per step, as vLLM's LLM._run_engine loop sees it)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap

import pytest

FAKE_VLLM = '''
import itertools
from types import SimpleNamespace

class SamplingParams:
    def __init__(self, temperature=0.0, seed=None, max_tokens=16, logprobs=None, prompt_logprobs=None, n=1, stop=None):
        self.temperature, self.seed, self.max_tokens, self.n, self.stop = temperature, seed, max_tokens, n, stop
        self.logprobs, self.prompt_logprobs = logprobs, prompt_logprobs

class Counter:
    def __init__(self):
        self.counter = 0
    def __next__(self):
        i = self.counter
        self.counter += 1
        return i

DRAWS = itertools.count()  # a random sample differs on every draw (seed None, temperature > 0)

class Engine:
    def __init__(self, kill_after=None):
        self.queue, self.steps, self.kill_after = [], 0, kill_after
    def has_unfinished_requests(self):
        return bool(self.queue)
    def step(self):
        if self.kill_after is not None and self.steps >= self.kill_after:
            raise KeyboardInterrupt("killed")
        self.steps += 1
        rid, prompt, sp = self.queue.pop(0)
        ids = prompt["prompt_token_ids"] if isinstance(prompt, dict) else None
        text = prompt if isinstance(prompt, str) else None
        body = f"{ids or text}|{sp.seed}" + (f"|draw{next(DRAWS)}" if sp.seed is None and sp.temperature > 0 else "")
        out = SimpleNamespace(index=0, text="gen:" + body, token_ids=[7, 8], finish_reason="stop", stop_reason=None,
                              cumulative_logprob=None, logprobs=None)
        return [SimpleNamespace(request_id=rid, prompt=text, prompt_token_ids=ids, outputs=[out], finished=True)]

class LLM:
    calls = []
    def __init__(self, kill_after=None):
        self.request_counter = Counter()
        self.llm_engine = Engine(kill_after)
    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        if not isinstance(prompts, list):
            prompts = [prompts]
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        LLM.calls.append(len(prompts))
        for p, sp in zip(prompts, sps):
            self.llm_engine.queue.append((str(next(self.request_counter)), p, sp))
        outs = []
        while self.llm_engine.has_unfinished_requests():
            outs += self.llm_engine.step()
        return sorted(outs, key=lambda o: int(o.request_id))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "__init__.py").write_text(textwrap.dedent(FAKE_VLLM))
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, m)
    from oellm.runner import gen_cache

    gen_cache = importlib.reload(gen_cache)
    assert gen_cache.install()
    import vllm

    vllm.LLM.calls.clear()
    monkeypatch.setenv("OELLM_GEN_CACHE", str(tmp_path / "gc" / "x.jsonl"))
    return vllm, gen_cache, tmp_path


def _prompts(n):
    return [{"prompt_token_ids": [i, i + 1]} for i in range(n)]


def test_a_kill_keeps_the_finished_generations_and_a_rerun_generates_the_rest(fake):
    vllm, gc, tmp = fake
    sp = vllm.SamplingParams(temperature=0.0, seed=0)
    full = [o.outputs[0].text for o in vllm.LLM().generate(_prompts(6), sp)]
    for f in (tmp / "gc").glob("x.jsonl.*"):
        f.unlink()
    with pytest.raises(KeyboardInterrupt):
        vllm.LLM(kill_after=4).generate(_prompts(6), sp)  # the job's time limit
    assert sum(1 for f in (tmp / "gc").glob("x.jsonl.*") for _ in f.open()) == 4
    vllm.LLM.calls.clear()
    again = vllm.LLM().generate(_prompts(6), sp)
    assert vllm.LLM.calls == [2], "only the two unfinished requests reach the engine"
    assert [o.outputs[0].text for o in again] == full and again[0].outputs[0].token_ids == [7, 8]
    assert again[0].finished and again[0].prompt_token_ids == [0, 1]


def test_the_stop_switch_ends_the_process_like_a_kill(fake, tmp_path):
    vllm, gc, tmp = fake
    (tmp / "stop_mod.py").write_text("import vllm\nvllm.LLM().generate([{'prompt_token_ids': [i]} for i in range(5)], "
                                     "vllm.SamplingParams(seed=0))\nprint('NOT REACHED')\n")
    env = {"PYTHONPATH": f"{tmp}:{gc.__file__.rsplit('/oellm/', 1)[0]}", "OELLM_GEN_CACHE": str(tmp / "gc" / "s.jsonl"),
           "OELLM_GEN_CACHE_STOP_AFTER": "3", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-m", "oellm.runner.gen_cache", "stop_mod"], cwd=tmp, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 124 and "NOT REACHED" not in r.stdout
    assert sum(1 for _ in next((tmp / "gc").glob("s.jsonl.*")).open()) == 3


def test_repeats_of_one_key_keep_their_own_samples(fake):
    vllm, gc, _ = fake
    sp = vllm.SamplingParams(temperature=1.0, seed=None)  # five draws of the same request
    first = [o.outputs[0].text for o in vllm.LLM().generate([{"prompt_token_ids": [1]}] * 5, sp)]
    assert len(set(first)) == 5
    vllm.LLM.calls.clear()
    again = [o.outputs[0].text for o in vllm.LLM().generate([{"prompt_token_ids": [1]}] * 5, sp)]
    assert vllm.LLM.calls == [] and sorted(again) == sorted(first)


def test_logprob_requests_and_other_parameters_are_not_cached(fake):
    vllm, gc, _ = fake
    vllm.LLM().generate(_prompts(2), vllm.SamplingParams(seed=0))
    vllm.LLM.calls.clear()
    vllm.LLM().generate(_prompts(2), vllm.SamplingParams(seed=0, prompt_logprobs=1))  # log-likelihood scoring
    vllm.LLM().generate(_prompts(2), vllm.SamplingParams(seed=0, max_tokens=99))      # another request
    assert vllm.LLM.calls == [2, 2]
    assert gc.key_of(_prompts(1)[0], vllm.SamplingParams(logprobs=5)) is None


def test_old_signature_text_prompts_and_no_cache_without_the_variable(fake, monkeypatch):
    vllm, gc, _ = fake
    sp = vllm.SamplingParams(seed=0)
    llm = vllm.LLM()
    one = llm.generate(prompt_token_ids=[[3, 4]], sampling_params=sp)
    assert one[0].outputs[0].text.startswith("gen:[3, 4]")
    vllm.LLM.calls.clear()
    assert vllm.LLM().generate(prompt_token_ids=[[3, 4]], sampling_params=sp)[0].outputs[0].text == one[0].outputs[0].text
    assert vllm.LLM.calls == []
    vllm.LLM().generate("hello", sp)
    vllm.LLM.calls.clear()
    assert vllm.LLM().generate("hello", sp).outputs[0].text.startswith("gen:hello")  # a single prompt in, one out
    monkeypatch.delenv("OELLM_GEN_CACHE")
    vllm.LLM().generate("hello", sp)
    assert vllm.LLM.calls == [1]


def test_the_module_entry_point_runs_a_module_with_the_cache(fake, tmp_path):
    vllm, gc, tmp = fake
    (tmp / "target_mod.py").write_text(textwrap.dedent('''
        import sys, vllm
        out = vllm.LLM().generate([{"prompt_token_ids": [9]}], vllm.SamplingParams(seed=1))
        print("OUT", sys.argv[1:], out[0].outputs[0].text, getattr(vllm.LLM.generate, "_oellm_gen_cache", False))
    '''))
    env = {"PYTHONPATH": f"{tmp}:{gc.__file__.rsplit('/oellm/', 1)[0]}", "OELLM_GEN_CACHE": str(tmp / "gc" / "m.jsonl"),
           "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-m", "oellm.runner.gen_cache", "target_mod", "--x", "1"], cwd=tmp, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "OUT ['--x', '1'] gen:[9]|1 True" in r.stdout
    assert json.loads(next((tmp / "gc").glob("m.jsonl.*")).read_text().splitlines()[0])["outputs"][0]["text"] == "gen:[9]|1"


def test_commands_give_each_call_its_own_cache(tmp_path):
    from oellm.runner import commands
    from oellm.runner.layout import Row
    from oellm.runner.plan import GroupSettings, SessionSettings, build_plan

    rows = [Row("vllm/AIME24/0", "AIME24", 0, "evalchemy"), Row("lighteval/xcopa:it/0", "xcopa:it", 0, "lighteval"),
            Row("vllm/gsm8k/4", "gsm8k", 4, "lm-eval-harness")]
    plan = build_plan(model_name="m", model_dir="/m", view_dir="/v", out=str(tmp_path), rows=rows,
                      lm_eval=SessionSettings(), lighteval=GroupSettings(), evalchemy=GroupSettings())
    by = {u.kind: u for u in plan.units}
    ec = commands.unit_commands(plan, by["evalchemy"], by["evalchemy"].rows)[0]
    assert ec.argv[1:4] == ["-m", "oellm.runner.gen_cache", "eval.eval"]
    assert ec.env["OELLM_GEN_CACHE"] == f"{plan.root}/gen_cache/evalchemy_AIME24_0.jsonl"
    le = commands.unit_commands(plan, by["lighteval"], by["lighteval"].rows)[0]
    assert le.env["OELLM_GEN_CACHE"].endswith("/gen_cache/lighteval_xcopa_it_0.jsonl")
    lm = commands.unit_commands(plan, by["lm_eval_session"], [], shard=3)[0]
    assert lm.env["OELLM_GEN_CACHE"].endswith("/gen_cache/lm_lm0_s3.jsonl")
