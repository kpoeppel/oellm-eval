# Node-local runner: plans, workers, runtimes

`oellm-eval schedule` renders a Slurm array and owns the allocation. The node-local runner does
not: something else hands it nodes (a Slurm array task, an oellm-autoexp pool job on MareNostrum 5
or JUPITER, a workstation) and it only runs the evals, writes the results, done markers and a
provenance record. Both clusters run the same plan, the same commands and the same results layout.

```
plan.json      what to run for one model: rows -> units, every setting   (oellm/runner/plan.py)
runtime.json   how a harness command runs on this machine                 (oellm/runner/runtime.py)
worker         python -m oellm.runner.worker --plan plan.json --runtime runtime.json --unit <id>
```

## Plan

```bash
python -m oellm.runner.cli plan --model-dir <hf model> --view-dir <identity view> --out <dir> \
    [--groups vllm=flag-evals-vllm,lighteval=flag-evals-lighteval] [--names ...] \
    [--settings settings.json] [--limit 20] --init-queue -o <dir>/<model>/launches/<id>/plan.json
python -m oellm.runner.cli status --plan plan.json
```

Rows come from the task groups (`<half>/<task>/<n_shot>`; `flag-evals-vllm` + `flag-evals-lighteval`
are the 438 FLAG rows). Units:

| kind | runs | model |
|---|---|---|
| `lm_eval_session` | one vLLM engine (`replicas` engines per node, each on `tp*dp` GPUs) evaluating the shared lm-eval queue until it is empty; long rows are split into document parts | plain |
| `lighteval` | its rows in one lighteval call (`backend`: `vllm` or `accelerate`), per-sample details saved | plain |
| `evalchemy` | its rows, one evalchemy call each | identity chat-template view |

lighteval must load the plain model: it applies any chat template it finds and then drops the tasks'
stop sequences. `settings.json` holds the engines and model-specific environment per harness
(`SessionSettings` / `GroupSettings` in plan.py).

## Runtime

`local` runs `python` of the current environment (MareNostrum 5: the node-local vLLM install with
the staged harnesses on `pythonpath`). `apptainer` starts one image per harness (`images`) per
command, binds the model, the results root, this checkout and `binds` (e.g. the patched HumanEval
grader) at their own paths, and passes the environment explicitly. This checkout is always first on
`PYTHONPATH`. `cleanup_globs`, `prune`, `hash_files` and `hang_minutes` cover node-local leftovers,
version clashes, provenance of patched files and the post-shutdown vLLM hang. `require` pins
package versions per harness (`--require lm-eval-harness:lm_eval=0.4.12`): the worker probes them
first and runs no command of a unit whose runtime differs (exit 3, recorded as `version_mismatch`).

An `lm_eval_session` with `dp > 1` runs one vLLM frontend per GPU (node_ep.py) on vLLM's offline
data parallelism, which has no DP coordinator: whenever one rank calls the engine, every rank must,
or the busy engine waits in its DP all-reduce until the 30-minute gloo timeout. lm-eval pads the
ranks to equal request counts; the response cache and the generation cache therefore reuse results
only when every rank has all of a call's requests cached (`node_ep.dp_all`), else every rank
evaluates all of its requests. Rolling log-likelihood rows (different window counts per rank) need
`dp 1`.

The runner imports only the standard library: a worker runs on a bare host `python3`.

## Worker

`--unit <id>` / `--index <i>` (a Slurm array task id) / `--all`; `--print` shows the commands. A
lighteval or evalchemy unit marks every row whose result is there; a failed call leaves its rows
unmarked (exit 1, or 0 with `--missing-ok` for callers that retry failed jobs). Each unit run writes
`<root>/provenance/<unit>__<host>__<utc>.json`: plan and runtime hashes, the code (this checkout's
commit, `$OELLM_CODE_STAMP`), package versions inside each runtime, hashes of `hash_files`, Slurm
ids, GPUs, every command with its exit status, rows done and missing.

## Results layout

```
<out>/<model>[_limit<l>]/
  results/, done/, parts/, queue_lm/      lm-eval
  lighteval/lighteval_<key>/ (+ details/), lighteval/done/
  evalchemy/evalchemy_<task>_<n>/, evalchemy/done/
  provenance/, launches/<id>/{plan.json, runtime.json, manifest.json}
```

## Reports

```bash
python -m oellm.reports.collect --root <out>/<model>=<model> --chrf-patched -o <model>.tasks.csv
python -m oellm.reports.check --csv <model>.tasks.csv --plan plan.json -o <model>.missing.csv
```

`collect` writes the CSV summarize-evals reads (the JUPITER collector's columns and rules; it also
reads `oellm-eval schedule` run dirs with `--runs`). `check` lists planned rows without a result.
`oellm.reports.manifest.write_manifest(launch_dir, plan, runtime, caller=...)` freezes a launch.
