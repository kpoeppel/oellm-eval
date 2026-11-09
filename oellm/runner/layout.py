"""Where a model's results live: the one definition of the results layout.

    <root> = <out>/<model name>[_limit<l>]
      results/<task>_<n_shot>/...          lm-eval (lm_eval_queue.py), samples_*.jsonl beside each result
      done/<task>_<n_shot>.ok              lm-eval row done (a part: <task>_<n_shot>@<k>of<K>.ok)
      parts/, response_cache/, summary_shard<i>.tsv
      queue_lm/{pending,running,failed}/   the lm-eval row queue (queue.py)
      lighteval/lighteval_<group key>/     one lighteval call; details/ = per-sample parquet
      lighteval/done/<key>_<n_shot>.ok
      evalchemy/evalchemy_<key>_<n_shot>/  one evalchemy call (its examples in the results json)
      evalchemy/done/<key>_<n_shot>.ok
      provenance/<unit>__<host>__<utc>.json   what ran, where, with which code (worker.py)
      launches/<id>/plan.json, runtime.json, manifest.json

A row is named ``<half>/<task>/<n_shot>`` (half: the task super group's suffix, e.g. ``vllm`` for
flag-evals-vllm); an lm-eval queue item may be a part of a row, ``<name>@<k>of<K>``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

HARNESSES = ("lm-eval-harness", "lighteval", "evalchemy")


def key_of(task: str) -> str:
    """The file key of a task name."""
    return re.sub(r"[/:| ]", "_", task)


@dataclass(frozen=True)
class Row:
    name: str     # <half>/<task>/<n_shot>
    task: str
    n_shot: int
    harness: str  # one of HARNESSES

    def __post_init__(self):
        if self.harness not in HARNESSES:
            raise ValueError(f"{self.name}: unknown harness {self.harness!r}")

    @staticmethod
    def parse_name(name: str) -> tuple[str, str, int]:
        """``vllm/arc_challenge/25`` -> (half, task, n_shot); the task may contain slashes."""
        half, rest = name.split("/", 1)
        task, n_shot = rest.rsplit("/", 1)
        return half, task, int(n_shot)


def results_root(out: str | Path, model_name: str, limit: float | int = 0) -> Path:
    return Path(out) / (model_name + (f"_limit{limit:g}" if limit else ""))


def group_key(rows: list[Row]) -> str:
    """The output dir key of one harness call: the row's own, or a hash of its rows (a rerun of
    the same rows writes to the same dir)."""
    if len(rows) == 1:
        return f"{key_of(rows[0].task)}_{rows[0].n_shot}"
    tasks = ",".join(r.task for r in rows)
    shots = ",".join(str(r.n_shot) for r in rows)
    return "group_" + hashlib.sha1(f"{tasks}\n{shots}\n".encode()).hexdigest()[:10]


class Layout:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def done_dir(self, harness: str) -> Path:
        return self.root / "done" if harness == "lm-eval-harness" else self.root / harness / "done"

    def marker(self, row: Row) -> Path:
        if row.harness == "lm-eval-harness":
            return self.done_dir(row.harness) / f"{row.task}_{row.n_shot}.ok"
        return self.done_dir(row.harness) / f"{key_of(row.task)}_{row.n_shot}.ok"

    def is_done(self, row: Row) -> bool:
        return self.marker(row).exists()

    def lighteval_out(self, rows: list[Row]) -> Path:
        return self.root / "lighteval" / f"lighteval_{group_key(rows)}"

    def evalchemy_out(self, row: Row) -> Path:
        return self.root / "evalchemy" / f"evalchemy_{key_of(row.task)}_{row.n_shot}"

    def gen_cache(self, name: str) -> Path:
        """A harness call's generation cache (gen_cache.py): <root>/gen_cache/<name>.jsonl[.<pid>]."""
        return self.root / "gen_cache" / f"{name}.jsonl"

    @property
    def queue_dir(self) -> Path:
        return self.root / "queue_lm"

    @property
    def provenance_dir(self) -> Path:
        return self.root / "provenance"

    @property
    def launches_dir(self) -> Path:
        return self.root / "launches"
