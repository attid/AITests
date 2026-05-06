from __future__ import annotations

import json
import re
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_eval.config import JsonExactTask, JsonSchemaTask, NumericTolerance


def extract_json(text: str) -> Any:
    """Parse JSON from raw model output. Handles raw JSON and ```json blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("no valid JSON found")


def _compare(
    expected: Any,
    actual: Any,
    *,
    tolerance: NumericTolerance,
    case_sensitive_strings: bool,
) -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        exp_dict = cast(dict[Any, Any], expected)
        act_dict = cast(dict[Any, Any], actual)
        if not exp_dict:
            return 1.0
        scores: list[float] = []
        for key, exp_val in exp_dict.items():
            if key not in act_dict:
                scores.append(0.0)
            else:
                scores.append(
                    _compare(
                        exp_val,
                        act_dict[key],
                        tolerance=tolerance,
                        case_sensitive_strings=case_sensitive_strings,
                    )
                )
        return sum(scores) / len(scores)

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0.0
        exp_list = cast(list[Any], expected)
        act_list = cast(list[Any], actual)
        if len(exp_list) != len(act_list):
            return 0.0
        if not exp_list:
            return 1.0
        return sum(
            _compare(e, a, tolerance=tolerance, case_sensitive_strings=case_sensitive_strings)
            for e, a in zip(exp_list, act_list, strict=True)
        ) / len(exp_list)

    if isinstance(expected, bool) or isinstance(actual, bool):
        return 1.0 if expected is actual else 0.0

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        bound = max(tolerance.abs, tolerance.rel * abs(float(expected)))
        return 1.0 if abs(float(expected) - float(actual)) <= bound else 0.0

    if expected is None:
        return 1.0 if actual is None else 0.0

    e_str = str(expected)
    a_str = str(actual)
    if not case_sensitive_strings:
        e_str = e_str.lower().strip()
        a_str = a_str.lower().strip()
    return 1.0 if e_str == a_str else 0.0


def score_json_exact(task: JsonExactTask, output: str) -> tuple[float, dict[str, Any]]:
    try:
        parsed = extract_json(output)
    except (ValueError, json.JSONDecodeError) as e:
        return 0.0, {
            "type": task.type,
            "json_ok": False,
            "notes": [f"json_parse_error:{e}"],
        }
    s = _compare(
        task.expected,
        parsed,
        tolerance=task.numeric_tolerance,
        case_sensitive_strings=task.case_sensitive_strings,
    )
    return s, {"type": task.type, "json_ok": True, "notes": []}


def score_json_schema(task: JsonSchemaTask, output: str) -> tuple[float, dict[str, Any]]:
    try:
        parsed = extract_json(output)
    except (ValueError, json.JSONDecodeError) as e:
        return 0.0, {
            "type": task.type,
            "json_ok": False,
            "notes": [f"json_parse_error:{e}"],
        }

    try:
        Draft202012Validator.check_schema(task.schema_)
    except SchemaError as e:
        return 0.0, {
            "type": task.type,
            "json_ok": True,
            "notes": [f"bad_schema:{e.message}"],
        }

    validator = Draft202012Validator(task.schema_)
    errors = list(validator.iter_errors(parsed))  # pyright: ignore[reportUnknownMemberType]
    if errors:
        msgs = ";".join(e.message for e in errors[:3])
        return 0.0, {
            "type": task.type,
            "json_ok": True,
            "notes": [f"schema_violation:{msgs}"],
        }

    if task.expected_values is None:
        return 1.0, {"type": task.type, "json_ok": True, "notes": []}

    # 0.5 for valid schema; +0.5 if expected_values match.
    if isinstance(parsed, dict):
        parsed_dict = cast(dict[Any, Any], parsed)
        all_match = all(
            key in parsed_dict and parsed_dict[key] == val
            for key, val in task.expected_values.items()
        )
    else:
        all_match = False
    return (1.0 if all_match else 0.5), {
        "type": task.type,
        "json_ok": True,
        "notes": [] if all_match else ["expected_values_mismatch"],
    }
