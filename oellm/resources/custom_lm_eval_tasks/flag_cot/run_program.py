"""Run one candidate program in code_eval's sandbox, in a fresh interpreter.

    python3 run_program.py <code_eval execute.py> <program file> <timeout>

Prints "passed" or the failure reason. utils._run_code starts one of these per candidate from a
thread pool: a fresh interpreter instead of os.fork(), which filelock's fork audit refuses while
another thread is forking ("os.fork is unsafe while filelock is changing descriptor ownership").
The program itself runs exactly as in code_eval: execute.unsafe_execute (reliability_guard, time
limit, swallowed stdout).
"""

import importlib.util
import sys


def main():
    execute_py, program_file, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
    spec = importlib.util.spec_from_file_location("code_eval_execute", execute_py)
    execute = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execute)
    result = []
    execute.unsafe_execute(open(program_file).read(), result, timeout)
    print(result[0] if result else "timed out", flush=True)


if __name__ == "__main__":
    main()
