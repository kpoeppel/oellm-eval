"""A work queue on a shared filesystem: workers claim items by an atomic rename.

    <queue>/pending/<order>_<item>.json   to do, the lowest order first (the longest first)
    <queue>/running/<order>_<item>.json   claimed (renamed out of pending/: one worker wins) and
                                          touched by its worker while it runs (the lease)
    <queue>/failed/<order>_<item>.json    gave up after ``max_attempts``

A claim not touched for ``lease_s`` (its worker died) goes back to pending/. Items are strings;
for lm-eval they are row names or parts of rows (``vllm/mmlu/5@1of4``, lm_eval_parts.py) and
``lm_item_done`` reads the runner's done markers.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Callable

from oellm.runner.layout import Layout, Row


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", name)


class Queue:
    def __init__(self, root: str | Path, lease_s: float = 600.0, max_attempts: int = 2):
        self.root = Path(root)
        self.lease_s, self.max_attempts = lease_s, max_attempts
        for d in ("pending", "running", "failed"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def _known(self) -> set[str]:
        out = set()
        for d in ("pending", "running", "failed"):
            for p in (self.root / d).glob("*.json"):
                try:
                    out.add(json.loads(p.read_text())["item"])
                except (OSError, ValueError, KeyError):
                    continue
        return out

    def init(self, items: list[tuple[str, float]], is_done: Callable[[str], bool] | None = None) -> int:
        """Add the items that are not done, queued, running or failed; the number added."""
        known = self._known()
        added = 0
        start = len(list((self.root / "pending").glob("*.json"))) + len(list((self.root / "running").glob("*.json")))
        for order, (item, secs) in enumerate(sorted(items, key=lambda x: -x[1]), start=start):
            if item in known or (is_done is not None and is_done(item)):
                continue
            path = self.root / "pending" / f"{order:05d}_{_safe(item)}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"item": item, "est_s": secs, "attempts": 0}))
            tmp.replace(path)
            added += 1
        return added

    def reclaim(self) -> int:
        """Claims not touched for lease_s back to pending (their worker is gone)."""
        n = 0
        now = time.time()
        for p in sorted((self.root / "running").glob("*.json")):
            try:
                if now - p.stat().st_mtime < self.lease_s:
                    continue
                os.rename(p, self.root / "pending" / p.name)
                n += 1
            except FileNotFoundError:
                continue  # done or reclaimed by another worker meanwhile
        return n

    def claim(self) -> tuple[Path, dict] | None:
        self.reclaim()
        for p in sorted((self.root / "pending").glob("*.json")):
            target = self.root / "running" / p.name
            try:
                os.rename(p, target)
            except FileNotFoundError:
                continue  # another worker took it
            os.utime(target)
            data = json.loads(target.read_text())
            data["worker"] = f"{socket.gethostname()}:{os.getpid()}"
            target.write_text(json.dumps(data))
            return target, data
        return None

    @staticmethod
    def touch(path: Path) -> None:
        try:
            os.utime(path)
        except FileNotFoundError:
            pass

    def complete(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    def fail(self, path: Path, data: dict) -> None:
        data["attempts"] = int(data.get("attempts", 0)) + 1
        dest = "failed" if data["attempts"] >= self.max_attempts else "pending"
        (self.root / dest / path.name).write_text(json.dumps(data))
        path.unlink(missing_ok=True)

    def requeue_failed(self) -> int:
        """Failed items back to pending with their attempts reset; the number moved."""
        n = 0
        for p in sorted((self.root / "failed").glob("*.json")):
            data = json.loads(p.read_text())
            data["attempts"] = 0
            (self.root / "pending" / p.name).write_text(json.dumps(data))
            p.unlink()
            n += 1
        return n

    def status(self) -> dict[str, int]:
        return {d: len(list((self.root / d).glob("*.json"))) for d in ("pending", "running", "failed")}


def lm_item_done(item: str, layout: Layout) -> bool:
    """An lm-eval item's done marker: the row's (it also covers the row's parts) or the part's own."""
    from oellm.runner.lm_eval_parts import parse, part_key

    base, k, n = parse(item)
    _, task, n_shot = Row.parse_name(base)
    done = layout.done_dir("lm-eval-harness")
    if (done / f"{task}_{n_shot}.ok").exists():
        return True
    return k is not None and (done / f"{part_key(task, n_shot, k, n)}.ok").exists()
