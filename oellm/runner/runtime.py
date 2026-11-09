"""How a harness command runs on this machine (runtime.json, written by whoever allocates the node).

    local       the current machine's Python environment (``python``): MN5's node-local vLLM
                install with the oellm-eval stage on PYTHONPATH
    apptainer   one image per harness (``images``), started per command: JUPITER's vLLM and
                lighteval images; the model, the results root, this oellm-eval checkout and
                ``binds`` are bound at their own paths, the environment is passed explicitly

Either way this checkout comes first on PYTHONPATH, so the harness modules (lm_eval_queue,
lighteval_cli) are the plan's. ``env`` / ``harness_env`` are set for every / one harness's
commands; after this checkout come ``harness_pythonpath`` (a harness's own pins) and ``pythonpath``.
``python`` may name an environment variable (``${INFER_PY}``), expanded where the command starts.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

OELLM_ROOT = str(Path(__file__).resolve().parents[2])  # the checkout (or site dir) holding oellm/


@dataclass
class Runtime:
    kind: str = "local"
    python: str = "python3"
    env: dict[str, str] = field(default_factory=dict)
    harness_env: dict[str, dict[str, str]] = field(default_factory=dict)
    pythonpath: list[str] = field(default_factory=list)
    harness_pythonpath: dict[str, list[str]] = field(default_factory=dict)
    dirs: dict[str, str] = field(default_factory=dict)  # cwd keys of commands, e.g. evalchemy
    images: dict[str, str] = field(default_factory=dict)
    apptainer: str = "apptainer"
    apptainer_args: list[str] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)
    harness_binds: dict[str, list[str]] = field(default_factory=dict)
    gpus: int = 4                       # GPUs of the node (lm-eval replicas split them)
    cleanup_globs: list[str] = field(default_factory=list)  # node-local leftovers, before and after a unit
    prune: dict[str, list[str]] = field(default_factory=dict)  # removed before a harness's first command
    hash_files: list[str] = field(default_factory=list)  # provenance: sha256 of these (patched harness files)
    hang_minutes: float = 15.0          # end a command this long silent after vLLM's shutdown (0: never)
    name: str = ""

    def check(self) -> None:
        if self.kind not in ("local", "apptainer"):
            raise ValueError(f"runtime kind {self.kind!r}")
        if self.kind == "apptainer":
            missing = [h for h in ("lm-eval-harness", "lighteval", "evalchemy") if h not in self.images]
            if missing:
                raise ValueError(f"apptainer runtime: no image for {missing}")

    @classmethod
    def from_json(cls, text: str) -> "Runtime":
        data = json.loads(text)
        unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"runtime: unknown keys {unknown}")
        rt = cls(**data)
        rt.check()
        return rt

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    # ------------------------------------------------------------ commands

    def _pythonpath(self, harness: str, inherited: str | None) -> str:
        # a harness's own pins (e.g. lighteval's metric packages) shadow the shared paths
        front = [OELLM_ROOT, *self.harness_pythonpath.get(harness, []), *self.pythonpath]
        return ":".join([*front, *([inherited] if inherited else [])])

    def wrap(self, argv: list[str], harness: str, env: dict[str, str], cwd_key: str | None,
             mounts: list[str]) -> tuple[list[str], dict[str, str], str | None]:
        """(argv, environment, cwd) to start a harness command; ``mounts``: paths the command
        reads or writes (bound at their own paths in an image)."""
        if argv[0] == "python":  # $VAR / ${VAR}: an interpreter known only on the node (a node-local venv)
            argv = [os.path.expandvars(self.python), *argv[1:]]
        cwd = self.dirs.get(cwd_key) if cwd_key else None
        if cwd_key and not cwd:
            raise ValueError(f"runtime has no directory {cwd_key!r} (dirs)")
        cmd_env = {**self.env, **self.harness_env.get(harness, {}), **env}
        if self.kind == "local":
            full = {**os.environ, **cmd_env}
            full["PYTHONPATH"] = self._pythonpath(harness, os.environ.get("PYTHONPATH"))
            return argv, full, cwd
        binds: list[str] = []
        for b in [*mounts, OELLM_ROOT, *self.binds, *self.harness_binds.get(harness, [])]:
            b = b if ":" in b else f"{b}:{b}"
            if b not in binds:
                binds.append(b)
        inner = {**cmd_env, "PYTHONPATH": self._pythonpath(harness, cmd_env.get("PYTHONPATH"))}
        if "CUDA_VISIBLE_DEVICES" in os.environ and "CUDA_VISIBLE_DEVICES" not in inner:
            inner["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        wrapped = [self.apptainer, "exec", *self.apptainer_args, "--bind", ",".join(binds)]
        if cwd:
            wrapped += ["--pwd", cwd]
        # `env K=V` inside the image: values may hold commas (apptainer --env splits at them)
        wrapped += [self.images[harness], "env", *[f"{k}={v}" for k, v in sorted(inner.items())], *argv]
        return wrapped, dict(os.environ), None

    # ------------------------------------------------------------ node housekeeping

    def cleanup(self) -> list[str]:
        removed = []
        for pattern in self.cleanup_globs:
            for f in glob.glob(os.path.expandvars(pattern)):
                try:
                    os.unlink(f)
                    removed.append(f)
                except OSError:
                    pass
        return removed

    def prune_for(self, harness: str) -> list[str]:
        removed = []
        for pattern in self.prune.get(harness, []):
            for d in glob.glob(os.path.expandvars(pattern)):
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
        return removed


def read_runtime(path: str | Path) -> Runtime:
    return Runtime.from_json(Path(path).read_text())
