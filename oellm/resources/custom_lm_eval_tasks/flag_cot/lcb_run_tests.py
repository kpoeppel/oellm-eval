"""Run one LiveCodeBench sample against its tests, in a fresh interpreter.

    python3 lcb_run_tests.py <livecodebench_utils.py> <request.json> <result.json> <timeout> <memory bytes>

request.json = {"tests": [...], "completion": str, "is_extracted": bool}; result.json = the list of
per-test verdicts (true / false) up to and including the first failure, like Evalchemy's runner.

The tests themselves are Evalchemy's (livecodebench_utils: prepare_test_input_output_*,
run_test_std / run_test_func, reliability_guard), loaded from the image by path. Two limits
Evalchemy's runner (grade_one.py -> run_tests_for_one_example) does not set:
  - a time limit PER TEST (`timeout` s, the 6 s it passes to lcb_run). Its runner only bounds the
    whole sample, (timeout + 1) x tests + 15 s, so a program that loops spends that entire budget
    -- minutes per sample, the bulk of livecodebench_cont's grading time (2026-10-08).
  - an address-space cap (RLIMIT_AS), as several samples run at once; a program that exceeds it
    gets a MemoryError and fails its test.
One interpreter per sample instead of Evalchemy's three (grade_one, a Manager, a spawned worker),
and without importing scipy: livecodebench_utils imports scipy.stats but never uses it.
"""

import copy
import importlib.util
import json
import resource
import signal
import sys
import types


class TestTimeout(BaseException):
    """BaseException: the sample's own `except Exception` must not swallow it."""


def _on_alarm(signum, frame):
    raise TestTimeout()


def main():
    utils_py, request_json, result_json = sys.argv[1:4]
    timeout, memory = float(sys.argv[4]), int(sys.argv[5])

    scipy = types.ModuleType("scipy")
    scipy.stats = types.ModuleType("scipy.stats")
    sys.modules.update({"scipy": scipy, "scipy.stats": scipy.stats})
    spec = importlib.util.spec_from_file_location("lcb_utils", utils_py)
    lcb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lcb)

    request = json.load(open(request_json))
    tests, completion, is_extracted = request["tests"], request["completion"], request["is_extracted"]
    out = open(result_json, "w")  # opened before reliability_guard
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    signal.signal(signal.SIGALRM, _on_alarm)
    lcb.reliability_guard()  # as Evalchemy's runner: disables os.kill, os.fork, subprocess, ...

    verdicts = []
    functional = tests[0]["testtype"] == "functional"
    for test in tests:
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            if functional:
                test_input, test_output = lcb.prepare_test_input_output_functional(test, is_extracted)
                passed, _ = lcb.run_test_func(
                    completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output)
                )
            else:
                test_input, test_output = lcb.prepare_test_input_output_std(test)
                passed, _ = lcb.run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
        except BaseException:  # TestTimeout, MemoryError, any error, and sys.exit() in the program
            passed = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        verdicts.append(bool(passed))
        if not passed:
            break
    sys.stdout = sys.__stdout__
    out.write(json.dumps(verdicts))
    out.close()


if __name__ == "__main__":
    main()
