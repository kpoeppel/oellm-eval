"""oellm.runner.node_ep: under node-local DP every rank calls the engine whenever one does -- the
offline DP engines have no coordinator, a rank that skips the engine hangs its peers. Two gloo ranks
on CPU with unequal caches: lm-eval's response cache (CachingLM) and oellm's generation cache."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(target, tmp_path, size=2):
    port = _port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_rank_main, args=(target, r, size, port, str(tmp_path), q)) for r in range(size)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in ps), [p.exitcode for p in ps]
    return dict(q.get() for _ in ps)


def _rank_main(target, rank, size, port, tmp, q):
    import torch.distributed as dist

    os.environ["OELLM_DP_RANK"] = str(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size)
    try:
        q.put((rank, globals()[target](rank, tmp)))
    finally:
        dist.destroy_process_group()


class _Req(SimpleNamespace):
    pass


SP = SimpleNamespace(temperature=0.0, seed=None, max_tokens=8, n=1, stop=None)  # greedy: cacheable
REC = {"index": 0, "text": "cached", "token_ids": [], "finish_reason": "stop", "stop_reason": None,
       "cumulative_logprob": None}


class _FakeLM:
    """Records each request-method call (a stand-in for lm-eval's VLLM: one call = engine calls)."""

    def __init__(self):
        self.calls = []

    def set_cache_hook(self, hook):
        pass

    def loglikelihood(self, requests):
        self.calls.append(len(requests))
        return [(-float(len(r.args[0])), False) for r in requests]


def _response_cache_rank(rank, tmp):
    from lm_eval.api.model import CachingLM

    from oellm.runner.node_ep import consistent_response_cache

    consistent_response_cache()
    reqs = [_Req(args=(f"ctx{rank}-{i}", " a")) for i in range(4)]
    db = os.path.join(tmp, "cache")
    # pre-fill: rank 0 has all of its requests cached, rank 1 three of four (a crash mid-row)
    warm = CachingLM(_FakeLM(), f"{db}_rank{rank}.db")
    warm.loglikelihood(reqs if rank == 0 else reqs[:3])
    lm = _FakeLM()
    first = CachingLM(lm, f"{db}_rank{rank}.db").loglikelihood(reqs)
    # now both ranks are complete: nobody calls the engine
    second = CachingLM(lm, f"{db}_rank{rank}.db").loglikelihood(reqs)
    return {"calls": lm.calls, "same": first == second, "n": len(first)}


def test_response_cache_all_ranks_call_the_engine(tmp_path):
    pytest.importorskip("lm_eval")
    pytest.importorskip("sqlitedict")
    res = _run("_response_cache_rank", tmp_path)
    # before the fix: rank 0 made no call ([]), rank 1 one call of 1 request -> the engines hung
    assert res[0]["calls"] == [4] and res[1]["calls"] == [4]
    assert all(r["same"] and r["n"] == 4 for r in res.values())


def _gen_cache_rank(rank, tmp):
    import json

    from oellm.runner import gen_cache

    path = os.path.join(tmp, f"gen_r{rank}")
    keys = [gen_cache.key_of(f"p{rank}-{i}", SP) for i in range(3)]
    stored = keys if rank == 0 else keys[:1]  # rank 0 fully cached, rank 1 one of three
    with open(path + ".1", "w") as f:
        for k in stored:
            f.write(json.dumps({"key": k, "outputs": [REC]}) + "\n")
    os.environ["OELLM_GEN_CACHE"] = path
    seen = []

    class LLM:
        def generate(self, prompts=None, sampling_params=None):
            seen.append(len(prompts))
            return [SimpleNamespace(outputs=[SimpleNamespace(text="new", token_ids=[])]) for _ in prompts]

    out = gen_cache._wrap(LLM.generate)(LLM(), [f"p{rank}-{i}" for i in range(3)], SP)
    return {"calls": seen, "texts": [o.outputs[0].text for o in out]}


def test_gen_cache_all_ranks_call_the_engine(tmp_path):
    res = _run("_gen_cache_rank", tmp_path)
    # before the fix: rank 0 skipped generate ([]) while rank 1 generated 2 prompts
    assert res[0]["calls"] == [3] and res[1]["calls"] == [3]
    assert res[0]["texts"] == ["new"] * 3


def _gen_cache_full_rank(rank, tmp):
    import json

    from oellm.runner import gen_cache

    path = os.path.join(tmp, f"genfull_r{rank}")
    with open(path + ".1", "w") as f:
        for i in range(2):
            f.write(json.dumps({"key": gen_cache.key_of(f"p{rank}-{i}", SP),
                                "outputs": [REC]}) + "\n")
    os.environ["OELLM_GEN_CACHE"] = path
    seen = []

    class LLM:
        def generate(self, prompts=None, sampling_params=None):
            seen.append(len(prompts))
            return []

    out = gen_cache._wrap(LLM.generate)(LLM(), [f"p{rank}-{i}" for i in range(2)], SP)
    return {"calls": seen, "texts": [o.outputs[0].text for o in out]}


def test_gen_cache_all_cached_everywhere_skips(tmp_path):
    res = _run("_gen_cache_full_rank", tmp_path)
    assert all(r["calls"] == [] and r["texts"] == ["cached"] * 2 for r in res.values())
