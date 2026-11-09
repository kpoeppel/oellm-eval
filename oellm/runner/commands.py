"""The harness commands of a plan's units: the one definition of how each harness is called.

A Command's argv starts with ``python`` -- the runtime's interpreter (runtime.py); ``cwd`` is a
runtime directory key (``evalchemy``: the evalchemy checkout) or None. Model directories:

    lm-eval     the plain model (lm-eval applies no chat template)
    lighteval   the plain model: lighteval applies any template it finds and then drops the tasks'
                stop sequences (FLORES generations ran on past the translation)
    evalchemy   the identity chat-template view (evalchemy applies the template)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oellm.runner.layout import Row
from oellm.runner.plan import Plan, Unit

ACCELERATE_BATCH_SIZE = 32  # oellm-eval's lighteval batch size off a local run (main.py)


@dataclass
class Command:
    argv: list[str]
    harness: str                 # which runtime image runs it
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None       # runtime directory key
    rows: list[Row] = field(default_factory=list)  # the rows whose results it writes


def lm_eval_session(plan: Plan, unit: Unit, shard: int) -> Command:
    g = unit.engine
    argv = ["python", "-m", "oellm.runner.lm_eval_queue", "--model", plan.model_dir, "--root", plan.root,
            "--queue", str(plan.layout.queue_dir), "--shard", str(shard), "--tp", str(g.tp), "--dp", str(g.dp),
            "--ep", "1" if g.expert_parallel else "0", "--gpu-mem", str(g.gpu_memory_utilization),
            "--max-num-seqs", str(g.max_num_seqs)]
    if g.max_num_batched_tokens is not None:
        argv += ["--max-num-batched-tokens", str(g.max_num_batched_tokens)]
    argv += ["--request-batch-size", str(g.request_batch_size) if g.request_batch_size else "auto"]
    if g.response_cache:
        argv.append("--response-cache")
    if plan.limit:
        argv += ["--limit", f"{plan.limit:g}"]
    env = {**unit.env, "OELLM_GEN_CACHE": str(plan.layout.gen_cache(f"lm_{unit.id}_s{shard}"))}
    return Command(argv, "lm-eval-harness", env=env)


def lighteval(plan: Plan, unit: Unit, rows: list[Row]) -> Command:
    g = unit.engine
    tasks = ",".join(f"{r.task}|{r.n_shot}" for r in rows)
    if unit.backend == "accelerate":  # HF transformers: oellm-eval's template.sbatch command
        model_args = f"model_name={plan.model_dir},trust_remote_code=True,batch_size={ACCELERATE_BATCH_SIZE}"
        if g.tp == 1:  # one GPU: HF's automatic map split even a 130M model (tied embeddings on the last GPU)
            model_args += ",model_parallel=False"
        env = dict(unit.env)
    else:
        model_args = (f"model_name={plan.model_dir},trust_remote_code=True,dtype=bfloat16,"
                      f"gpu_memory_utilization={g.gpu_memory_utilization},max_num_seqs={g.max_num_seqs},")
        if g.max_num_batched_tokens is not None:
            model_args += f"max_num_batched_tokens={g.max_num_batched_tokens},"
        if g.max_model_length is not None:
            model_args += f"max_model_length={g.max_model_length},"
        model_args += f"tensor_parallel_size={g.tp}"
        env = {**unit.env, "OELLM_EP": "1" if g.expert_parallel else "0"}  # read by the patched lighteval
    if unit.backend == "vllm":
        env["OELLM_GEN_CACHE"] = str(plan.layout.gen_cache(plan.layout.lighteval_out(rows).name))
    argv = ["python", "-m", "oellm.runner.lighteval_cli", unit.backend, model_args, tasks,
            "--load-tasks-multilingual", "--output-dir", str(plan.layout.lighteval_out(rows)), "--save-details"]
    if plan.limit:
        argv += ["--max-samples", f"{plan.limit:g}"]
    return Command(argv, "lighteval", env=env, rows=list(rows))


def evalchemy(plan: Plan, unit: Unit, row: Row) -> Command:
    g = unit.engine
    model_args = (f"pretrained={plan.view_dir},trust_remote_code=True,dtype=bfloat16,"
                  f"gpu_memory_utilization={g.gpu_memory_utilization},max_num_seqs={g.max_num_seqs},"
                  f"tensor_parallel_size={g.tp}")
    if "OELLM_VLLM_ENFORCE_EAGER" in unit.env:  # the model code's switch; else vLLM's default
        model_args += f",enforce_eager={unit.env['OELLM_VLLM_ENFORCE_EAGER'] == '1'}"
    if g.dp > 1:
        model_args += f",data_parallel_size={g.dp},data_parallel_backend=mp"
    if g.expert_parallel:
        model_args += ",enable_expert_parallel=True"
    if g.max_model_length is not None:
        model_args += f",max_model_len={g.max_model_length}"
    out = plan.layout.evalchemy_out(row)
    # through the generation cache: a rerun after a kill keeps every finished generation
    argv = ["python", "-m", "oellm.runner.gen_cache", "eval.eval", "--model", "vllm", "--tasks", row.task,
            "--model_args", model_args, "--batch_size", "auto", "--output_path", str(out)]
    if plan.limit:
        argv += ["--limit", f"{plan.limit:g}"]
    env = {**unit.env, "OELLM_GEN_CACHE": str(plan.layout.gen_cache(out.name))}
    return Command(argv, "evalchemy", env=env, cwd="evalchemy", rows=[row])


def unit_commands(plan: Plan, unit: Unit, rows: list[Row], shard: int = 0) -> list[Command]:
    """The commands of a unit for its rows still to do (lm-eval sessions: the queue decides)."""
    if unit.kind == "lm_eval_session":
        return [lm_eval_session(plan, unit, shard)]
    if not rows:
        return []
    if unit.kind == "lighteval":
        return [lighteval(plan, unit, rows)]
    # one call per task: evalchemy writes all results at the end, so one task's crash would lose
    # a whole group
    return [evalchemy(plan, unit, r) for r in rows]
