"""Stop and resume generations: every finished vLLM generation is appended to a cache file the
moment it completes; a rerun of the same command takes the stored ones and generates only the rest.

Harness calls write their results only at the end (evalchemy after all repeats of a benchmark,
lighteval after all its tasks), so a job ended by its time limit lost everything it generated --
LiveCodeBench is 3066 generations of up to 32k tokens. With ``OELLM_GEN_CACHE=<file>`` set,
``install()`` wraps ``vllm.LLM.generate`` (any vLLM version with ``LLM.request_counter`` and
``LLM.llm_engine.step``):

- a request's key is its prompt (token ids or text) and its sampling parameters; a key requested
  several times (repeats with seed None) keeps several outputs, each handed out once
- requests that ask for log-probs (``logprobs`` / ``prompt_logprobs``: log-likelihood scoring) pass
  through uncached
- outputs are appended while the engine runs (one JSON line per finished request; one file per
  process, ``<file>.<pid>``), so a kill keeps everything finished before it
- the cached outputs are returned as plain objects with the fields harnesses read (``outputs[i].
  text / token_ids / finish_reason / stop_reason / cumulative_logprob``, ``prompt``,
  ``prompt_token_ids``, ``finished``)

    python -m oellm.runner.gen_cache <module> [args ...]   # runpy the module with the cache installed

(the worker runs evalchemy this way; lighteval_cli and lm_eval_queue install it themselves).
``OELLM_GEN_CACHE_STOP_AFTER=N`` (tests): end the process after N stored generations, as a kill.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import runpy
import sys
from types import SimpleNamespace

SP_FIELDS = ("n", "temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "min_tokens", "stop",
             "stop_token_ids", "ignore_eos", "repetition_penalty", "presence_penalty", "frequency_penalty",
             "skip_special_tokens", "spaces_between_special_tokens", "include_stop_str_in_output",
             "truncate_prompt_tokens", "bad_words")
_installed = False


def _prompt_repr(prompt):
    if isinstance(prompt, str):
        return {"text": prompt}
    if isinstance(prompt, dict):
        if prompt.get("prompt_token_ids") is not None:
            return {"ids": list(prompt["prompt_token_ids"])}
        if prompt.get("prompt") is not None:
            return {"text": prompt["prompt"]}
        return None
    if isinstance(prompt, (list, tuple)) and all(isinstance(t, int) for t in prompt):
        return {"ids": list(prompt)}
    return None


def _sp_repr(sp):
    out = {}
    for f in SP_FIELDS:
        v = getattr(sp, f, None)
        out[f] = list(v) if isinstance(v, (set, tuple)) else v
    return out


def key_of(prompt, sp) -> str | None:
    """The cache key of one request, or None when it must not be cached."""
    if sp is None or getattr(sp, "logprobs", None) is not None or getattr(sp, "prompt_logprobs", None) is not None:
        return None
    p = _prompt_repr(prompt)
    if p is None:
        return None
    return hashlib.sha256(json.dumps([p, _sp_repr(sp)], sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.stored: dict[str, list[dict]] = {}
        for f in sorted(glob.glob(glob.escape(path) + ".*")):
            with open(f, errors="replace") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a line cut by a kill
                    self.stored.setdefault(rec["key"], []).append(rec)
        self._fh = None
        self.added = 0

    def has_all(self, keys: list) -> bool:
        """Every key has a stored generation (repeats of a key need as many)."""
        need: dict[str, int] = {}
        for k in keys:
            if k is None:
                return False
            need[k] = need.get(k, 0) + 1
        return all(len(self.stored.get(k, ())) >= n for k, n in need.items())

    def take(self, key: str) -> dict | None:
        lst = self.stored.get(key)
        return lst.pop(0) if lst else None

    def add(self, key: str, output) -> None:
        if self._fh is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fh = open(f"{self.path}.{os.getpid()}", "a", buffering=1)
        rec = {"key": key, "outputs": [
            {"index": getattr(o, "index", i), "text": o.text, "token_ids": list(getattr(o, "token_ids", []) or []),
             "finish_reason": getattr(o, "finish_reason", None), "stop_reason": getattr(o, "stop_reason", None),
             "cumulative_logprob": getattr(o, "cumulative_logprob", None)}
            for i, o in enumerate(output.outputs)]}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        self.added += 1
        stop = int(os.environ.get("OELLM_GEN_CACHE_STOP_AFTER", "0") or 0)
        if stop and self.added >= stop:
            print(f"[gen_cache] OELLM_GEN_CACHE_STOP_AFTER={stop}: stopping as a kill would", flush=True)
            os._exit(124)


def _restore(rec: dict, prompt) -> SimpleNamespace:
    p = _prompt_repr(prompt) or {}
    outs = [SimpleNamespace(index=o["index"], text=o["text"], token_ids=o["token_ids"], logprobs=None,
                            finish_reason=o["finish_reason"], stop_reason=o["stop_reason"],
                            cumulative_logprob=o["cumulative_logprob"]) for o in rec["outputs"]]
    return SimpleNamespace(request_id=f"cached-{rec['key'][:12]}", prompt=p.get("text"), prompt_token_ids=p.get("ids"),
                           prompt_logprobs=None, outputs=outs, finished=True)


def _wrap(original):
    def generate(self, prompts=None, sampling_params=None, *args, **kwargs):
        path = os.environ.get("OELLM_GEN_CACHE")
        if not path:
            return original(self, prompts, sampling_params, *args, **kwargs)
        if prompts is None and kwargs.get("prompt_token_ids") is not None:  # old vLLM signature
            prompts = [{"prompt_token_ids": ids} for ids in kwargs.pop("prompt_token_ids")]
        single = isinstance(prompts, (str, dict)) or (
            isinstance(prompts, (list, tuple)) and len(prompts) > 0 and all(isinstance(t, int) for t in prompts))
        plist = [prompts] if single else list(prompts)
        sps = sampling_params if isinstance(sampling_params, (list, tuple)) else [sampling_params] * len(plist)
        if len(sps) != len(plist):
            return original(self, prompts, sampling_params, *args, **kwargs)
        cache = getattr(self, "_oellm_gen_cache", None)
        if cache is None or cache.path != path:
            cache = self._oellm_gen_cache = Cache(path)
        keys = [key_of(p, sp) for p, sp in zip(plist, sps)]
        results: list = [None] * len(plist)
        todo = []
        # node-local EP frontends: reuse only when every rank's call is fully cached, else every rank
        # generates all of its prompts -- a rank skipping the engine hangs its peers (node_ep.dp_all)
        use = True
        if os.environ.get("OELLM_DP_RANK") is not None:
            from oellm.runner.node_ep import dp_all

            use = dp_all(cache.has_all(keys))
        for i, k in enumerate(keys):
            rec = cache.take(k) if k and use else None
            if rec is not None:
                results[i] = _restore(rec, plist[i])
            else:
                todo.append(i)
        reused = len(plist) - len(todo)
        print(f"[gen_cache] {reused} of {len(plist)} generation(s) from {path}", flush=True)
        if todo:
            counter = getattr(self, "request_counter", None)
            engine = getattr(self, "llm_engine", None)
            start = getattr(counter, "counter", None)
            watch = {}
            if start is not None and engine is not None:
                watch = {str(start + j): (keys[i], _prompt_repr(plist[i])) for j, i in enumerate(todo) if keys[i]}
                step = engine.step

                def step_and_store(*a, **kw):
                    outs = step(*a, **kw)
                    for out in outs:
                        w = watch.get(getattr(out, "request_id", None))
                        if w is None or not getattr(out, "finished", False):
                            continue
                        expect = w[1]
                        got_ids = getattr(out, "prompt_token_ids", None)
                        if expect.get("ids") is not None and got_ids is not None and list(got_ids) != expect["ids"]:
                            continue  # not the request we predicted: never store a wrong answer
                        if expect.get("text") is not None and getattr(out, "prompt", None) not in (None, expect["text"]):
                            continue
                        cache.add(w[0], out)
                    return outs

                engine.step = step_and_store
            try:
                new = original(self, [plist[i] for i in todo], [sps[i] for i in todo], *args, **kwargs)
            finally:
                if watch:
                    engine.step = step
            for i, out in zip(todo, new):
                results[i] = out
        return results[0] if single else results

    generate._oellm_gen_cache = True
    return generate


def install() -> bool:
    """Wrap vllm.LLM.generate (once); False when vLLM is not importable."""
    global _installed
    if _installed:
        return True
    try:
        import vllm
    except ImportError:
        return False
    if not getattr(vllm.LLM.generate, "_oellm_gen_cache", False):
        vllm.LLM.generate = _wrap(vllm.LLM.generate)
    _installed = True
    return True


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: python -m oellm.runner.gen_cache <module> [args ...]")
    if os.environ.get("OELLM_GEN_CACHE"):
        install()
    sys.argv = [args[0], *args[1:]]
    runpy.run_module(args[0], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
