"""lighteval's ``vllm`` / ``accelerate`` commands without Typer's command line.

Some Typer versions (0.27.3, MN5) drop the second positional argument of lighteval's commands, so
this calls ``lighteval.main_vllm.vllm`` / ``lighteval.main_accelerate.accelerate`` directly with
their option defaults resolved:

    python -m oellm.runner.lighteval_cli vllm|accelerate <model_args> <tasks> \\
        [--load-tasks-multilingual] [--output-dir DIR] [--save-details] [--max-samples N]

``accelerate`` is oellm-eval's JUPITER command (HF transformers); ``vllm`` serves models the HF path
cannot hold. Either way the model must be the plain one: lighteval applies any chat template it
finds, and with one it drops the tasks' stop sequences (vllm backend).
"""

from __future__ import annotations

import inspect
import sys

USAGE = ("usage: python -m oellm.runner.lighteval_cli vllm|accelerate <model_args> <tasks> "
         "[--load-tasks-multilingual] [--output-dir DIR] [--save-details] [--max-samples N]")


def _defaults(fn) -> dict:
    import typer

    out = {}
    for name, param in inspect.signature(fn).parameters.items():
        default = param.default
        if isinstance(default, typer.models.OptionInfo):
            out[name] = default.default
        elif default is not inspect.Parameter.empty:
            out[name] = default
    return out


def parse(args: list[str]) -> tuple[str, str, str, dict]:
    """(backend, model_args, tasks, options) of the command line; unknown flags are errors."""
    if len(args) < 3 or args[0] not in ("vllm", "accelerate"):
        raise SystemExit(USAGE)
    backend, model_args, tasks = args[0], args[1], args[2]
    opts: dict = {}
    i = 3
    while i < len(args):
        arg = args[i]
        if arg == "--load-tasks-multilingual":
            opts["load_tasks_multilingual"] = True
            i += 1
        elif arg == "--save-details":
            opts["save_details"] = True
            i += 1
        elif arg == "--output-dir":
            opts["output_dir"] = args[i + 1]
            i += 2
        elif arg == "--max-samples":
            opts["max_samples"] = int(args[i + 1])
            i += 2
        else:
            raise SystemExit(f"unknown argument {arg!r}\n{USAGE}")
    return backend, model_args, tasks, opts


def main(argv: list[str] | None = None) -> int:
    backend, model_args, tasks, opts = parse(sys.argv[1:] if argv is None else argv)
    if backend == "vllm":
        from lighteval.main_vllm import vllm as command
    else:
        from lighteval.main_accelerate import accelerate as command
    kwargs = _defaults(command)
    kwargs.pop("model_args", None)  # the positionals are passed explicitly, never as defaults
    kwargs.pop("tasks", None)
    kwargs.update(opts)
    command(model_args, tasks, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
