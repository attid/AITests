from llm_eval.scoring.preprocess import contains_forbidden, strip_thinking


def test_strip_thinking_basic() -> None:
    assert strip_thinking("<think>plan</think>answer") == "answer"


def test_strip_thinking_multiline() -> None:
    text = "<think>\nstep 1\nstep 2\n</think>\nfinal"
    assert strip_thinking(text).strip() == "final"


def test_strip_thinking_no_block() -> None:
    assert strip_thinking("plain") == "plain"


def test_strip_thinking_multiple_blocks() -> None:
    assert strip_thinking("<think>a</think>X<think>b</think>Y") == "XY"


def test_forbidden_word_boundary_russian() -> None:
    # "да" must NOT match inside "однажды"
    assert not contains_forbidden("однажды было", ["да"], word_boundary=True, case_sensitive=False)
    # but should match "да" as a word
    assert contains_forbidden("ну да, конечно", ["да"], word_boundary=True, case_sensitive=False)


def test_forbidden_substring_mode() -> None:
    assert contains_forbidden("правда", ["да"], word_boundary=False, case_sensitive=False)


def test_forbidden_case_sensitivity() -> None:
    assert contains_forbidden("Hello YES", ["yes"], word_boundary=True, case_sensitive=False)
    assert not contains_forbidden("Hello YES", ["yes"], word_boundary=True, case_sensitive=True)
