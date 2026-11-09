"""A plan: what to run for one model -- its rows split into units of work, with every setting.

    lm_eval_session   one engine (``replicas`` copies, each on ``tp * dp`` GPUs) that evaluates
                      the lm-eval queue's items until the queue is empty; all session units of a
                      plan share the queue (``lm_items``: rows, long rows split into parts)
    lighteval         its rows in one lighteval call (one model load)
    evalchemy         its rows, one evalchemy call each (evalchemy writes results only at the end)

The plan is plain JSON (``Plan.to_json`` / ``Plan.from_json``, unknown keys are errors) and holds no
machine-specific paths besides the model's and the results root: how a command runs here is the
runtime's (runtime.py). Rows come from oellm-eval's task groups (``rows_from_task_groups``) or a
caller's list; rows already done (layout markers) are left out at planning time.
"""

from __future__ import annotations

import dataclasses
import heapq
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from oellm.runner.layout import Layout, Row, results_root

PLAN_VERSION = 1
UNIT_KINDS = ("lm_eval_session", "lighteval", "evalchemy")
LOAD_S = 600.0  # one model load, inside a reference row time


@dataclass
class Engine:
    """One vLLM engine: ``tp`` x ``dp`` GPUs (dp > 1: node-local DP frontends, expert parallel)."""

    tp: int = 4
    dp: int = 1
    expert_parallel: bool = True
    max_num_seqs: int = 64
    max_num_batched_tokens: int | None = 32768
    max_model_length: int | None = None
    gpu_memory_utilization: float = 0.9
    request_batch_size: int | None = None  # lm-eval: None = auto
    response_cache: bool = False           # lm-eval: per-request SQLite cache (needs a batch size)
    replicas: int = 1                      # lm-eval sessions: engines per node, each on its own GPUs

    def check(self, where: str) -> None:
        if self.tp < 1 or self.dp < 1 or self.replicas < 1:
            raise ValueError(f"{where}: tp, dp and replicas must be positive")
        if self.dp > 1 and not self.expert_parallel:
            raise ValueError(f"{where}: node-local DP needs expert_parallel")
        if self.response_cache and self.request_batch_size is None:
            raise ValueError(f"{where}: response_cache needs request_batch_size")


@dataclass
class Unit:
    id: str
    kind: str
    rows: list[Row] = field(default_factory=list)
    engine: Engine = field(default_factory=Engine)
    env: dict[str, str] = field(default_factory=dict)  # model-specific switches (e.g. OELLM_*)
    backend: str = "vllm"                               # lighteval: vllm | accelerate
    est_s: float = 0.0

    def check(self) -> None:
        if self.kind not in UNIT_KINDS:
            raise ValueError(f"unit {self.id}: unknown kind {self.kind!r}")
        if self.backend not in ("vllm", "accelerate") or (self.backend == "accelerate" and self.kind != "lighteval"):
            raise ValueError(f"unit {self.id}: backend {self.backend!r} (accelerate: lighteval only)")
        want = {"lm_eval_session": "lm-eval-harness", "lighteval": "lighteval", "evalchemy": "evalchemy"}[self.kind]
        bad = [r.name for r in self.rows if r.harness != want]
        if bad:
            raise ValueError(f"unit {self.id}: rows of another harness: {bad}")
        self.engine.check(f"unit {self.id}")


@dataclass
class Plan:
    model_name: str
    model_dir: str        # the plain model: lm-eval, lighteval (it applies any chat template it finds)
    view_dir: str         # the identity chat-template view: evalchemy (falls back to model_dir)
    root: str             # results root (layout.py)
    rows: list[Row]
    units: list[Unit]
    lm_items: list[list] = field(default_factory=list)  # [item, est_s], longest first
    limit: float = 0
    protocol: str = "v0.01"
    task_groups: list[str] = field(default_factory=list)
    oellm_rev: str = ""
    created_utc: str = ""
    version: int = PLAN_VERSION

    @property
    def layout(self) -> Layout:
        return Layout(self.root)

    def unit(self, uid: str) -> Unit:
        for u in self.units:
            if u.id == uid:
                return u
        raise KeyError(f"no unit {uid!r} in the plan (units: {[u.id for u in self.units]})")

    def check(self) -> None:
        if self.version != PLAN_VERSION:
            raise ValueError(f"plan version {self.version}, this runner reads {PLAN_VERSION}")
        ids = [u.id for u in self.units]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate unit ids: {ids}")
        for u in self.units:
            u.check()
        if self.lm_items and not any(u.kind == "lm_eval_session" for u in self.units):
            raise ValueError("lm-eval items but no lm_eval_session unit")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        plan = _load(cls, json.loads(text), "plan")
        plan.check()
        return plan

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)
        return path


def read_plan(path: str | Path) -> Plan:
    return Plan.from_json(Path(path).read_text())


def _load(cls, data, where: str):
    """A dataclass from JSON, strictly: unknown keys are errors."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    kw = {}
    for name, value in data.items():
        if cls is Plan and name == "rows" or cls is Unit and name == "rows":
            kw[name] = [_load(Row, r, f"{where}.{name}") for r in value]
        elif cls is Plan and name == "units":
            kw[name] = [_load(Unit, u, f"{where}.units") for u in value]
        elif cls is Unit and name == "engine":
            kw[name] = _load(Engine, value, f"{where}.engine")
        else:
            kw[name] = value
    return cls(**kw)


# ---------------------------------------------------------------- rows


def rows_from_task_groups(halves: dict[str, str]) -> list[Row]:
    """Rows of oellm-eval task (super) groups: {half: group}, e.g. {"vllm": "flag-evals-vllm",
    "lighteval": "flag-evals-lighteval"}; names ``<half>/<task>/<n_shot>``."""
    from oellm.task_groups import _expand_task_groups  # needs the tool's dependencies (pyyaml)

    rows, seen = [], set()
    for half, group in halves.items():
        for r in _expand_task_groups([group]):
            name = f"{half}/{r.task}/{int(r.n_shot)}"
            if name in seen:
                raise ValueError(f"duplicate row {name}")
            seen.add(name)
            rows.append(Row(name=name, task=str(r.task), n_shot=int(r.n_shot), harness=r.suite))
    return rows


# ---------------------------------------------------------------- planning


@dataclass
class SessionSettings:
    """lm-eval: ``workers`` session units over one queue; rows longer than ``split_s`` engine
    seconds become up to ``max_parts`` document parts."""

    workers: int = 1
    split_s: float = 300.0
    max_parts: int = 8
    engine: Engine = field(default_factory=lambda: Engine(tp=1, dp=4, max_num_seqs=256, request_batch_size=256))
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class GroupSettings:
    """lighteval / evalchemy: rows in ``groups`` units (at most ``max_rows`` rows each), balanced
    longest first by the reference row times."""

    groups: int = 1
    max_rows: int | None = None
    exclude_tasks: list[str] = field(default_factory=list)
    engine: Engine = field(default_factory=Engine)
    env: dict[str, str] = field(default_factory=dict)
    backend: str = "vllm"


def split_items(names: list[str], times: dict[str, float], split_s: float, max_parts: int) -> list[list]:
    """[item, expected seconds], longest first; a row longer than ``split_s`` becomes
    ceil(t / split_s) parts (at most ``max_parts``)."""
    med = statistics.median(times.values()) if times else 120.0
    items = []
    for name in names:
        t = float(times.get(name, med))
        k = min(max_parts, max(1, math.ceil(t / split_s))) if split_s > 0 else 1
        if k == 1:
            items.append([name, t])
        else:
            items.extend([f"{name}@{i}of{k}", t / k] for i in range(k))
    return sorted(items, key=lambda x: -x[1])


def expected_seconds(rows: list[Row], times: dict[str, float]) -> list[float]:
    """Per row: its reference job time (key ``<harness>|<task>|<n_shot>``) minus a load, else its
    harness's median."""
    key = lambda r: f"{r.harness}|{r.task}|{r.n_shot}"  # noqa: E731
    known: dict[str, list[float]] = {}
    for k, t in times.items():
        known.setdefault(k.split("|", 1)[0], []).append(max(60.0, t - LOAD_S))
    med = {s: sorted(v)[len(v) // 2] for s, v in known.items()}
    return [max(60.0, times[key(r)] - LOAD_S) if key(r) in times else med.get(r.harness, 600.0) for r in rows]


def balance(secs: list[float], groups: int) -> list[list[int]]:
    """Longest first onto the least loaded group; each group's members in input order."""
    heap = [(0.0, g) for g in range(groups)]
    out: list[list[int]] = [[] for _ in range(groups)]
    for s, i in sorted(((s, i) for i, s in enumerate(secs)), reverse=True):
        load, g = heapq.heappop(heap)
        out[g].append(i)
        heapq.heappush(heap, (load + s, g))
    return [sorted(g) for g in out if g]


def build_plan(*, model_name: str, model_dir: str, view_dir: str | None, out: str, rows: list[Row],
               lm_eval: SessionSettings | None = None, lighteval: GroupSettings | None = None,
               evalchemy: GroupSettings | None = None, limit: float = 0, protocol: str = "v0.01",
               task_groups: list[str] | None = None, oellm_rev: str = "",
               engine_times: dict[str, float] | None = None, row_times: dict[str, float] | None = None,
               tag: str = "") -> Plan:
    """The plan of ``rows`` for one model; rows with a done marker under the results root are left
    out. A harness whose settings are None is not planned. Unit ids: ``lm<i>``, ``le<i>``,
    ``ec<i>`` (+ ``_<tag>``)."""
    root = results_root(out, model_name, limit)
    layout = Layout(root)
    todo = [r for r in rows if not layout.is_done(r)]
    sfx = f"_{tag}" if tag else ""
    units: list[Unit] = []
    items: list[list] = []
    lm_rows = [r for r in todo if r.harness == "lm-eval-harness"]
    if lm_eval is not None and lm_rows:
        # a test run (limit) evaluates the first documents of each row: no parts
        items = split_items([r.name for r in lm_rows], engine_times or {}, 0 if limit else lm_eval.split_s,
                            lm_eval.max_parts)
        per = sum(s for _, s in items) / max(1, lm_eval.workers)
        units += [Unit(id=f"lm{i}{sfx}", kind="lm_eval_session", engine=lm_eval.engine, env=dict(lm_eval.env),
                       est_s=LOAD_S + per) for i in range(lm_eval.workers)]
    for harness, short, cfg in (("lighteval", "le", lighteval), ("evalchemy", "ec", evalchemy)):
        h_rows = [r for r in todo if r.harness == harness and (cfg is None or r.task not in cfg.exclude_tasks)]
        if cfg is None or not h_rows:
            continue
        secs = expected_seconds(h_rows, row_times or {})
        n = max(cfg.groups, math.ceil(len(h_rows) / cfg.max_rows) if cfg.max_rows else 1)
        for g, members in enumerate(balance(secs, min(n, len(h_rows))), start=1):
            loads = len(members) if harness == "evalchemy" else 1  # evalchemy: one call per task
            units.append(Unit(id=f"{short}{g}{sfx}", kind=harness, rows=[h_rows[i] for i in members],
                              engine=cfg.engine, env=dict(cfg.env), backend=cfg.backend,
                              est_s=loads * LOAD_S + sum(secs[i] for i in members)))
    plan = Plan(model_name=model_name, model_dir=str(model_dir), view_dir=str(view_dir or model_dir),
                root=str(root), rows=list(rows), units=units, lm_items=items, limit=limit, protocol=protocol,
                task_groups=list(task_groups or []), oellm_rev=oellm_rev,
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    plan.check()
    return plan


def init_queue(plan: Plan, lease_s: float = 600.0, max_attempts: int = 2) -> dict:
    """Queue the plan's lm-eval items that are not done, queued, running or failed (idempotent)."""
    from oellm.runner.queue import Queue, lm_item_done

    q = Queue(plan.layout.queue_dir, lease_s=lease_s, max_attempts=max_attempts)
    added = q.init([tuple(i) for i in plan.lm_items], lambda item: lm_item_done(item, plan.layout))
    return {"added": added, "items": len(plan.lm_items), **q.status()}
