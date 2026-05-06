import llm_eval


def test_package_imports() -> None:
    assert llm_eval.__version__ == "0.1.0"
