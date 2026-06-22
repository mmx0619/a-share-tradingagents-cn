"""统一结构化输出调用层。

这个文件解决一个核心稳定性问题：

    大模型输出不一定每次都是合法 JSON。

对于普通自然语言报告，这不是致命问题。
但对于 Router / Research Manager / Trader / Portfolio Manager 这类节点，
程序后续流转依赖固定字段。

所以这里统一做：

    1. 调用模型；
    2. 提取 assistant content；
    3. 用 Pydantic schema 校验；
    4. 失败后把错误反馈给模型重试；
    5. 多次失败后返回保守兜底对象。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from tradingagents_cn.llm.deepseek_client import extract_assistant_message


T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredOutputResult(Generic[T]):
    """结构化输出调用结果。"""

    value: T
    raw_text: str
    attempts: int
    used_fallback: bool
    errors: list[str]
    messages: list[dict[str, Any]]


def call_structured_output(
    llm_client: Any,
    messages: list[dict[str, Any]],
    schema_model: type[T],
    fallback_factory: Callable[[str], T],
    temperature: float = 0.0,
    max_retries: int = 2,
) -> StructuredOutputResult[T]:
    """调用大模型并返回经过 Pydantic 校验的对象。

    max_retries:
        解析失败后的重试次数。
        例如 max_retries=2 表示最多调用 3 次：
            第一次正常调用；
            失败后重试 2 次。
    """
    working_messages = [dict(message) for message in messages]
    schema_text = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
    errors: list[str] = []
    raw_text = ""

    for attempt in range(1, max_retries + 2):
        response = llm_client.chat(
            messages=working_messages,
            temperature=temperature,
        )
        assistant_message = extract_assistant_message(response)
        raw_text = str(assistant_message.get("content") or "")
        working_messages.append(assistant_message)

        try:
            value = parse_structured_text(raw_text, schema_model)
            return StructuredOutputResult(
                value=value,
                raw_text=raw_text,
                attempts=attempt,
                used_fallback=False,
                errors=errors,
                messages=working_messages,
            )
        except Exception as error:
            error_text = f"第 {attempt} 次结构化解析失败：{error}"
            errors.append(error_text)
            if attempt > max_retries:
                break

            working_messages.append(
                {
                    "role": "user",
                    "content": build_retry_prompt(
                        schema_text=schema_text,
                        error_text=error_text,
                    ),
                }
            )

    fallback = fallback_factory("；".join(errors) or "模型没有返回合法结构化结果。")
    return StructuredOutputResult(
        value=fallback,
        raw_text=raw_text,
        attempts=max_retries + 1,
        used_fallback=True,
        errors=errors,
        messages=working_messages,
    )


def parse_structured_text(text: str, schema_model: type[T]) -> T:
    """从模型文本中提取 JSON 并用 Pydantic 校验。"""
    json_text = extract_json_object_text(text)
    data = json.loads(json_text)
    try:
        return schema_model.model_validate(data)
    except ValidationError:
        raise


def extract_json_object_text(text: str) -> str:
    """从文本中提取 JSON 对象字符串。"""
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced_match:
        return fenced_match.group(1)

    object_match = re.search(r"\{.*\}", stripped, re.S)
    if object_match:
        return object_match.group(0)

    raise ValueError("模型输出中没有找到 JSON 对象。")


def build_retry_prompt(schema_text: str, error_text: str) -> str:
    """构造结构化输出失败后的重试提示。"""
    return (
        "你上一条回复没有通过程序校验。\n"
        f"错误信息：{error_text}\n\n"
        "请重新输出，并且只输出一个合法 JSON 对象，不要 Markdown，不要解释文字。\n"
        "必须符合下面的 JSON Schema：\n"
        f"{schema_text}"
    )
