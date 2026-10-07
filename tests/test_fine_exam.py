"""The fine code exam: partial credit by tests, valid Python, likelihood of the reference solutions."""

from evo.learning import fine_exam as fe
from evo.learning.code_tasks import CodeTask

TASK = CodeTask("add_3", "add_3(x)", "Return x plus 3.", ["assert add_3(1) == 4", "assert add_3(0) == 3", "assert add_3(-3) == 0"],
                "    return x + 3", 1)


def test_points_for_valid_python_and_for_every_test():
    right, half, wrong, broken = "    return x + 3", "    return 4 if x == 1 else 0", "    return x", "    return x +"
    assert fe.compiles(TASK, right) and not fe.compiles(TASK, broken) and not fe.compiles(TASK, "")
    assert fe.tests_passed(TASK, right) == 3 and fe.tests_passed(TASK, half) == 2 and fe.tests_passed(TASK, wrong) == 0
    assert fe.tests_passed(TASK, "    while True:\n        pass") == 0 and fe.tests_passed(TASK, "    import os\n    return x + 3") == 0
    g = fe.grade([TASK], {TASK.key: half})
    assert g == {"score": round(20 + 80 * 2 / 3, 2), "tests_passed": 2, "tests": 3, "solved": 0, "valid": 1, "tasks": 1}
    assert fe.grade([TASK], {TASK.key: right})["score"] == 100.0 and fe.grade([TASK], {TASK.key: right})["solved"] == 1
    assert fe.grade([TASK], {TASK.key: wrong})["score"] == 20.0 and fe.grade([TASK], {})["score"] == 0.0


def test_the_exam_is_the_same_held_out_set_as_the_judges():
    tasks = fe.exam_tasks()
    assert len(tasks) == 79 and all(t.pool == "exam" for t in tasks) and sum(len(t.tests) for t in tasks) > 150
    assert [t.key for t in fe.exam_tasks([tasks[0].key, tasks[5].key])] == [tasks[0].key, tasks[5].key]
    solved = fe.grade(tasks[:6], {t.key: t.solution for t in tasks[:6]})
    assert solved["score"] == 100.0 and solved["solved"] == 6 and solved["tests_passed"] == solved["tests"]
    assert "code score 100.0/100" in fe.text("m", {**solved, "nll": 0.5})
