"""RULER length rows of the lm-eval queue runner: "<task>__L<length>" names (lm_eval_queue.ruler_row)."""

from oellm.runner.lm_eval_queue import ruler_row


def test_ruler_length_rows_parse():
    assert ruler_row("niah_single_2__L65536") == ("niah_single_2", 65536)
    assert ruler_row("ruler_vt__L4096") == ("ruler_vt", 4096)


def test_other_tasks_are_not_ruler_rows():
    for task in ("niah_single_2", "coqa", "squadv2", "a__Lx", "__L4096", "mmlu__L"):
        assert ruler_row(task) is None, task
