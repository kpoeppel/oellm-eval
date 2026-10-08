"""Pre-training variants of the FLAG reasoning and code evals: `_cot` and `_cont` tasks.

Why. The release evals of these tasks (Evalchemy, 0-shot, greedy) measure the RESPONSE MODE of a
base model more than its ability: whether it opens a worked derivation, answers directly or stops.
Forcing the derivation with a " <think>\\n" prefill makes checkpoints that differ by 0.2-0.3 in the
release numbers score alike (oellm-autoexp docs/anneal-math/README.md, 2026-10-07).

  _cot   competition math, science QA, polymath: the release prompt (same problems, same wording)
         with " <think>\\n" appended, so generation starts inside a worked derivation; sampled.
  _cont  code as a continuation, the base-model protocol (Codex; Qwen/DeepSeek/Olmo base reports):
         the model continues a function stub / code block instead of answering an instruction.

All tasks sample n completions per problem (lm-eval `repeats`, kept by a take_first_k filter) and
report pass@1 (mean over the n samples) and pass@k with the unbiased estimator of Chen et al. 2021
(arXiv:2107.03374, Eq. 1). The settings and their sources are in each task YAML.

Context: our checkpoints have a 4096-token context. lm-eval's vLLM backend LEFT-truncates a prompt
when prompt + max_gen_toks exceeds it (maybe_truncate(..., shrink_gen_toks=False)), cutting the
start of the problem. This module, imported only when one of these tasks is loaded (one task per
lm-eval run in the FLAG launcher), switches the backend to shrink the generation budget instead,
so each prompt gets the rest of the context: max_gen_toks 4096 in a YAML means "4096 - this
prompt", the per-prompt budget of the math_cot runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb

import datasets

EVALCHEMY = "/opt/evalchemy"
BENCH = f"{EVALCHEMY}/eval/chat_benchmarks"
THINK = " <think>\n"  # the opener of the models' own worked solutions (docs/anneal-math)


def _shrink_budget_instead_of_truncating_prompt():
    try:
        import lm_eval.models.vllm_causallms as vllm_lm
    except Exception:  # another backend: nothing to patch
        return
    original = vllm_lm.maybe_truncate
    if getattr(original, "_flag_cot", False):
        return

    def maybe_truncate(*args, **kwargs):
        kwargs["shrink_gen_toks"] = True
        return original(*args, **kwargs)

    maybe_truncate._flag_cot = True
    vllm_lm.maybe_truncate = maybe_truncate


_shrink_budget_instead_of_truncating_prompt()


def _hub_cache():
    # Evalchemy loads its HF datasets with cache_dir=$HF_HUB_CACHE; the FLAG prefetch filled that
    # cache with exactly these calls, and compute nodes are offline.
    return os.environ.get("HF_HUB_CACHE")


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021, Eq. 1)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _pass_metrics(correct: list[float], ks: list[int]) -> dict[str, float]:
    """pass@1 is the mean score over the samples (partial credit allowed, e.g. JEEBench);
    pass@k (k > 1) counts a sample as correct only when it is fully correct."""
    n = len(correct)
    out = {"pass@1": sum(correct) / n}
    c = sum(1 for x in correct if x >= 1.0)
    for k in ks:
        if k > 1:
            out[f"pass@{k}"] = pass_at_k(n, c, k)
    return out


def _single(dataset_list: list[dict]) -> datasets.DatasetDict:
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(dataset_list)})


# --------------------------------------------------------------------------------------------
# Competition math: AIME24 / AIME25 / AMC23 / MATH500 (Evalchemy data and prompt)
# --------------------------------------------------------------------------------------------
MATH_PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
MATH_DATA = {
    "AIME24": f"{BENCH}/AIME24/data/aime24.json",
    "AIME25": f"{BENCH}/AIME25/data/aime25.json",
    "AMC23": f"{BENCH}/AMC23/data/amc23.json",
    "MATH500": f"{BENCH}/MATH500/data/math500.jsonl",
}


def load_math(source: str, **_) -> datasets.DatasetDict:
    """The Evalchemy data file of one task (rows: problem, answer, as in math_cot/math_sample.py)."""
    txt = open(MATH_DATA[source]).read().strip()
    try:
        rows = json.loads(txt)
        rows = rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
    return _single(
        [
            dict(
                problem=r.get("problem", r.get("question")),
                answer=str(r.get("expected_answer", r.get("answer"))),
                source=source,
            )
            for r in rows
        ]
    )


def math_doc_to_text(doc: dict) -> str:
    return MATH_PROMPT.format(problem=doc["problem"]) + THINK


_lenient = _load_path("flag_cot_lenient", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lenient.py"))


def _math_correct(source: str, answer: str, text: str) -> bool:
    """Evalchemy's rule (last \\boxed{}, hendrycks is_equiv; MATH500: Evalchemy's
    extract_math_answer) OR the lenient extraction (boxed -> answer phrase -> last number) --
    exactly the acc_lenient of oellm-autoexp scripts/downstream_eval/math_cot/grade.py."""
    from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

    try:
        if source == "MATH500":
            if EVALCHEMY not in sys.path:
                sys.path.insert(0, EVALCHEMY)
            from eval.utils.parsers import extract_math_answer

            pred = extract_math_answer(text)
        else:
            pred = remove_boxed(last_boxed_only_string(text))
    except Exception:
        pred = ""
    if is_equiv(str(answer), pred):
        return True
    return _lenient.lenient_ok(answer, text)


def math_process_results(doc: dict, results: list) -> dict[str, float]:
    samples = results[0]
    correct = [float(_math_correct(doc["source"], doc["answer"], s)) for s in samples]
    out = _pass_metrics(correct, [4, 16, 32])
    out = {k: v for k, v in out.items() if int(k.split("@")[1]) <= len(samples)}
    out["think_closed"] = sum("</think>" in s for s in samples) / len(samples)
    return out


# --------------------------------------------------------------------------------------------
# JEEBench (Evalchemy prompt library and compute_score)
# --------------------------------------------------------------------------------------------
_jee = None


def _jeebench():
    global _jee
    if _jee is None:
        _jee = _load_path("flag_cot_jeebench_utils", f"{BENCH}/JEEBench/utils.py")
    return _jee


JEE_PROMPTS = {  # Evalchemy JEEBench PROMPT_LIBRARY after prompt_for_boxed_answer()
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer."
    " Mark your solution, which should be exactly one multiple-choice letter, with \\boxed\nAnswer:",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer."
    " Mark your solution, which should be one or more multiple-choice letter(s), with \\boxed\nAnswer:",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer."
    " Mark your solution with \\boxed\nAnswer:",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer."
    " Mark your solution with \\boxed\nAnswer:",
}


def load_jeebench(**_) -> datasets.DatasetDict:
    ds = datasets.load_dataset("daman1209arora/jeebench", cache_dir=_hub_cache())
    return datasets.DatasetDict({"test": ds["test"]})


def jeebench_doc_to_text(doc: dict) -> str:
    # Evalchemy format_message(): prefix + "\n\n" + "Problem: " + question
    question = doc["question"].replace("\n\n", "\n").strip()
    return (JEE_PROMPTS[doc["type"]] + "\n\n" + "Problem: " + question).strip() + THINK


def jeebench_process_results(doc: dict, results: list) -> dict[str, float]:
    jee = _jeebench()
    scores = []
    for s in results[0]:
        try:
            resp = jee.remove_boxed(jee.last_boxed_only_string(s))
        except Exception:
            resp = None
        scores.append(float(jee.compute_score(doc["gold"], resp, doc["type"])))
    out = _pass_metrics(scores, [4])
    out["think_closed"] = sum("</think>" in s for s in results[0]) / len(results[0])
    return out


# --------------------------------------------------------------------------------------------
# GPQA Diamond (Evalchemy prompt, option shuffle with random.Random(42) in dataset order)
# --------------------------------------------------------------------------------------------
GPQA_PROMPT = (
    "Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.\n"
    "Problem: {problem}\n"
    "Options: {options}\n"
    "Answer:"
)


def load_gpqa_diamond(**_) -> datasets.DatasetDict:
    import random

    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", cache_dir=_hub_cache())
    rnd = random.Random(42)  # Evalchemy: one generator, shuffled in dataset order
    rows = []
    for ex in ds["train"]:
        answers = [ex["Correct Answer"], ex["Incorrect Answer 1"], ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        rnd.shuffle(answers)
        letters = ["A", "B", "C", "D"]
        options = ", ".join(f"{l}) {a}" for l, a in zip(letters, answers))
        gold = letters[answers.index(ex["Correct Answer"])]
        rows.append(dict(question=ex["Question"], options=options, gold=gold))
    return _single(rows)


def gpqa_doc_to_text(doc: dict) -> str:
    return GPQA_PROMPT.format(problem=doc["question"], options=doc["options"]) + THINK


_gpqa_mc = None


def _gpqa_answer(text: str) -> str:
    """Evalchemy GPQADiamond testing_utils.get_multiple_choice_answer."""
    global _gpqa_mc
    if _gpqa_mc is None:
        _gpqa_mc = _load_path("flag_cot_gpqa_testing_utils", f"{BENCH}/GPQADiamond/testing_utils.py")
    try:
        return _gpqa_mc.get_multiple_choice_answer(text)
    except Exception:
        return ""


def gpqa_process_results(doc: dict, results: list) -> dict[str, float]:
    correct = [float(_gpqa_answer(s) == doc["gold"]) for s in results[0]]
    out = _pass_metrics(correct, [])  # pass@k on a 4-way choice is inflated by guessing: pass@1 only
    out["think_closed"] = sum("</think>" in s for s in results[0]) / len(results[0])
    return out


# --------------------------------------------------------------------------------------------
# PolyMath (the existing polymath task's prompt, data and judge)
# --------------------------------------------------------------------------------------------
_poly = None


def _polymath():
    global _poly
    if _poly is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _poly = _load_path("flag_cot_polymath_utils", os.path.join(here, "..", "polymath", "utils.py"))
    return _poly


def polymath_doc_to_text(doc: dict) -> str:
    return _polymath().doc_to_text(doc) + THINK


def polymath_doc_to_target(doc: dict) -> str:
    return doc["answer"]


def polymath_process_results(doc: dict, results: list) -> dict[str, float]:
    """The polymath judge (math_equal on a \\boxed{} span), but on the LAST boxed span: inside a
    derivation the first one can be an intermediate result."""
    poly = _polymath()
    correct = []
    for s in results[0]:
        spans = poly.extract_boxed_content(s)
        pred = spans[-1] if spans else None
        correct.append(float(bool(poly.math_equal(pred, str(doc["answer"])))))
    out = _pass_metrics(correct, [4])
    out["think_closed"] = sum("</think>" in s for s in results[0]) / len(results[0])
    return out


# --------------------------------------------------------------------------------------------
# Code: shared execution through HF `code_eval` (as lm-eval's own humaneval / mbpp tasks)
# --------------------------------------------------------------------------------------------
_code_eval_execute = None
_RUN_PROGRAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_program.py")


def _run_code(programs: list[str], test: str, timeout: float = 3.0) -> list[bool]:
    """Run each candidate program + `test`; True where it passes.

    Execution is HF code_eval's (execute.unsafe_execute: reliability_guard, time limit), each
    program in a fresh interpreter (run_program.py) started from a thread pool. Not code_eval's
    own check_correctness: it os.fork()s from worker threads, which filelock's fork audit refuses
    while another thread is forking ("os.fork is unsafe while filelock is changing descriptor
    ownership"; mbpp_cont / humaneval_cont, 2026-10-08). code_eval's execute.py is read from the
    HF modules cache by path; the metric is not instantiated.
    """
    import glob
    import subprocess
    import tempfile

    global _code_eval_execute
    if _code_eval_execute is None:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        found = sorted(glob.glob(f"{hf_home}/modules/evaluate_modules/metrics/evaluate-metric--code_eval/*/execute.py"))
        if not found:
            raise FileNotFoundError(f"code_eval's execute.py is not cached under {hf_home}/modules")
        _code_eval_execute = found[-1]

    def run(program: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(program + "\n" + test)
        try:
            out = subprocess.run(
                [sys.executable, _RUN_PROGRAM, _code_eval_execute, fh.name, str(timeout)],
                capture_output=True,
                text=True,
                timeout=timeout + 30,
            ).stdout.strip()
            return out.endswith("passed")
        except subprocess.TimeoutExpired:
            return False
        finally:
            os.unlink(fh.name)

    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(run, programs))


# HumanEval (Evalchemy's humaneval-python.jsonl; classic completion: continue the stub)
def load_humaneval(**_) -> datasets.DatasetDict:
    rows = [json.loads(x) for x in open(f"{BENCH}/HumanEval/data/humaneval-python.jsonl") if x.strip()]
    return _single([dict(task_id=r["task_id"], prompt=r["prompt"], test=r["test"]) for r in rows])


def humaneval_doc_to_text(doc: dict) -> str:
    return doc["prompt"]


def humaneval_process_results(doc: dict, results: list) -> dict[str, float]:
    # The test code ends with check(<entry point>); the program is stub + continuation.
    passed = _run_code([doc["prompt"] + s for s in results[0]], doc["test"])
    return _pass_metrics([float(p) for p in passed], [16])


# MBPP (lm-eval's mbpp prompt and 3 fixed few-shot examples)
def mbpp_fewshot_samples():
    from lm_eval.tasks.mbpp.utils import list_fewshot_samples

    return list_fewshot_samples()


def mbpp_process_results(doc: dict, results: list) -> dict[str, float]:
    passed = _run_code(list(results[0]), "\n".join(doc["test_list"]))
    return _pass_metrics([float(p) for p in passed], [16])


# LiveCodeBench (Evalchemy data and test runner; continuation of a code block)
_lcb = None


def _livecodebench():
    global _lcb
    if _lcb is None:
        _lcb = _load_path("flag_cot_lcb_utils", f"{BENCH}/LiveCodeBench/livecodebench_utils.py")
    return _lcb


def load_livecodebench(**_) -> datasets.DatasetDict:
    lcb = _livecodebench()
    ds = datasets.load_dataset(
        "livecodebench/code_generation_lite",
        name="release_latest",
        version_tag="release_v2",
        split="test",
        trust_remote_code=True,
        cache_dir=_hub_cache(),
    )
    rows = []
    for row in ds:
        ex = lcb.map_to_example(
            {**row, "private_test_cases": lcb.translate_private_test_cases(row["private_test_cases"])}
        )
        ex["test"] = json.dumps(ex["test"])  # nested test cases: keep the dataset flat
        rows.append(ex)
    return _single(rows)


LCB_IMPORTS = (
    "from typing import *\nfrom collections import *\nfrom math import *\nfrom heapq import *\n"
    "from bisect import *\nfrom itertools import *\nfrom functools import *\nimport sys\n"
)
LCB_STDIN = "Read the inputs from stdin, solve the problem and write the answer to stdout."
LCB_FUNCTIONAL = "Complete the method of the given starter code."


def livecodebench_doc_to_text(doc: dict) -> str:
    starter = doc["entry_point"] or ""
    head = (
        "### Question\n" + doc["prompt"].strip() + "\n\n"
        "### Format\n" + (LCB_STDIN if doc["is_stdin"] else LCB_FUNCTIONAL) + "\n\n"
        "### Answer\n```python\n"
    )
    return head + (starter if not doc["is_stdin"] else "")


def _lcb_method_name(starter: str) -> str | None:
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter)
    return m.group(1) if m else None


def _lcb_grade(doc: dict, text: str) -> float:
    lcb = _livecodebench()
    problem = dict(doc)
    problem["test"] = json.loads(doc["test"])
    starter = doc["entry_point"] or ""
    method = _lcb_method_name(starter) if not doc["is_stdin"] else None
    code = text.split("```")[0]  # the continuation ends at the closing fence
    # Starter code uses typing names (List, Optional) and solutions use the usual standard
    # library; nothing here contains a "(", so the runner's function-name parse is unaffected.
    code = LCB_IMPORTS + code
    if not doc["is_stdin"]:
        code = starter + code
        if method:
            # Evalchemy's runner calls the TOP-LEVEL function named by the first "(" of the
            # program (here the starter's method name); expose the method under that name.
            code += f"\n\ndef {method}(*args, **kwargs):\n    return Solution().{method}(*args, **kwargs)\n"
    try:
        res = lcb.lcb_run(problem, lcb.post_process_code(code), 6, not doc["is_stdin"])
    except lcb.LCBInfrastructureError:
        raise
    except Exception:
        return 0.0
    return float(bool(res) and all(r[0] for r in res))


class LiveCodeBenchGrade:
    """Filter: replace every sample of every problem by its verdict (1.0 / 0.0), graded at once.

    Grading inside process_results (called once per problem) kept at most n samples in flight and
    waited for each problem's slowest one (6 s per private test): 511 x 8 samples took more than
    80 min and the task hit its 1:30 limit (v2anneal_120k, 2026-10-08). As a filter it sees all
    samples, so one pool stays busy. Each sample still runs in its own fresh interpreters
    (lcb_run -> grade_one.py), so the verdicts are unchanged.

    Not more workers: every sample starts three interpreters that import scipy from the container
    image, and at 64 workers squashfuse could not keep up -- graders overran their budget (an
    LCBInfrastructureError, 2026-10-08). A sample whose grader still overruns is retried once,
    alone, before the error is raised.
    """

    def __init__(self, **_):
        pass

    def apply(self, resps, docs):
        import threading

        lcb = _livecodebench()
        alone = threading.Lock()

        def grade(job):
            try:
                return _lcb_grade(*job)
            except lcb.LCBInfrastructureError:
                with alone:
                    return _lcb_grade(*job)

        resps = [list(samples) for samples in resps]  # take_first_k hands over a one-shot map
        jobs = [(d, text) for d, samples in zip(docs, resps) for text in samples]
        workers = int(os.environ.get("FLAG_LCB_WORKERS", "16"))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = iter(list(pool.map(grade, jobs)))
        return [[next(verdicts) for _ in samples] for samples in resps]


def livecodebench_process_results(doc: dict, results: list) -> dict[str, float]:
    # results[0] = this problem's verdicts (LiveCodeBenchGrade, after take_first_k)
    return _pass_metrics([float(v) for v in results[0]], [4])
