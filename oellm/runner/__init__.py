"""Node-local evaluation runner: oellm-eval runs evals on nodes it is handed, it does not allocate them.

    plan        rows -> units of work (an lm-eval engine session over a row queue, a lighteval
                group, an evalchemy group), written as plan.json (plan.py)
    runtime     how a harness command is executed on this machine: the current Python
                environment, or an apptainer image per harness (runtime.py)
    worker      ``python -m oellm.runner.worker --plan plan.json --runtime runtime.json --unit <id>``:
                runs one unit (or all) on the GPUs of the current allocation, writes done markers
                and a provenance record (worker.py)

The allocation is the caller's: a Slurm array task, a pool job (oellm-autoexp), a local machine.
Everything here imports only the standard library, so a worker runs on a bare host ``python3``
(and starts the harness images itself); the harness modules (lm_eval_queue.py, lighteval_cli.py)
run inside the harness runtime.

Results layout (one root per model and limit, layout.py): lm-eval under results/ and done/,
lighteval under lighteval/, evalchemy under evalchemy/, provenance/ and launches/ beside them.
"""
