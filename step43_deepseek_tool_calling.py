"""第 43 步：真实 DeepSeek Tool Calling 教学版。

前面第 31 到第 39 步，我们一直在用 mock 模拟 Tool Calling。

这一文件开始做真实模型版本：

用户问题
  ↓
发送给 DeepSeek，并附带 tools 定义
  ↓
DeepSeek 返回 tool_calls
  ↓
Python 程序执行本地工具函数
  ↓
把工具结果作为 tool 消息发回 DeepSeek
  ↓
DeepSeek 生成最终回答

注意：

1. 真正执行工具的仍然是 Python。
2. DeepSeek 只是返回“我要调用哪个工具、参数是什么”。
3. API Key 只从环境变量读取，不写进代码。
4. 本文件里的工具仍然是模拟工具，不联网取真实行情。
5. 下一步第 44 步再把工具换成真实 AKShare。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


@dataclass
class ToolExecutionResult:
    """工具执行结果。"""

    tool_call_id: str
    tool_name: str
    content: str


def get_env(name: str) -> str | None:
    """读取环境变量。

    这里不把 API Key 写进代码。

    程序只读取：

    DEEPSEEK_API_KEY

    如果你在 Windows 系统变量里配置了，
    Conda 环境启动后一般就能读到。
    """
    value = os.environ.get(name)
    if value:
        return value
    return None


def get_realtime_quote(symbol: str) -> str:
    """本地模拟行情工具。

    真实项目里，这个函数后面会替换成 AKShare 行情工具。

    当前第 43 步先不要引入真实数据源，
    只验证真实 DeepSeek Tool Calling 闭环。
    """
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def build_tools_schema() -> list[dict[str, Any]]:
    """构造发给 DeepSeek 的 tools schema。

    tools schema 的作用：

    告诉模型：

    你现在可以调用一个工具，
    工具名叫 get_realtime_quote，
    它需要一个参数 symbol。

    模型如果决定调用工具，
    会在返回结果里生成 tool_calls。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_realtime_quote",
                "description": "查询 A 股股票的实时行情快照。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "6 位 A 股股票代码，例如 002361。",
                        }
                    },
                    "required": ["symbol"],
                },
            },
        }
    ]


def call_deepseek_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    force_tool_name: str | None = None,
) -> dict[str, Any]:
    """调用 DeepSeek Chat Completions API。

    参数说明：

    - messages：对话消息。
    - tools：可用工具定义。
    - force_tool_name：是否强制模型调用某个工具。

    为什么这里允许 force_tool_name？

    教学时我们希望稳定看到 tool_calls。
    如果完全交给 auto，模型有时可能直接回答，不一定调用工具。

    所以本文件默认强制调用 get_realtime_quote，
    这样你每次运行都能看到真实 tool_calls。
    """
    api_key = get_env("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请先配置环境变量。")

    payload: dict[str, Any] = {
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "messages": messages,
        "temperature": 0.2,
    }

    if tools is not None:
        payload["tools"] = tools
        if force_tool_name:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": force_tool_name},
            }
        else:
            payload["tool_choice"] = "auto"

    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_assistant_message(response_json: dict[str, Any]) -> dict[str, Any]:
    """从 DeepSeek 返回结果里取出 assistant message。"""
    choices = response_json.get("choices")
    if not choices:
        raise ValueError("DeepSeek 返回结果中没有 choices。")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("DeepSeek 返回结果中没有合法 message。")

    return message


def execute_tool_call(tool_call: dict[str, Any]) -> ToolExecutionResult:
    """执行 DeepSeek 返回的一个 tool_call。

    DeepSeek 返回的 tool_call 大致长这样：

    {
        "id": "...",
        "type": "function",
        "function": {
            "name": "get_realtime_quote",
            "arguments": "{\"symbol\":\"002361\"}"
        }
    }

    注意：
    function.arguments 通常是 JSON 字符串，
    所以需要 json.loads() 解析。
    """
    tool_call_id = str(tool_call.get("id", ""))
    function_info = tool_call.get("function", {})
    tool_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", "{}")

    if not tool_call_id:
        raise ValueError("tool_call 缺少 id。")
    if tool_name != "get_realtime_quote":
        raise ValueError(f"不支持的工具名：{tool_name}")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ValueError(f"工具参数不是合法 JSON：{raw_arguments}") from error

    symbol = arguments.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError(f"工具参数缺少合法 symbol：{arguments}")

    content = get_realtime_quote(symbol)
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=content,
    )


def run_deepseek_tool_calling_demo() -> str:
    """运行真实 DeepSeek Tool Calling 闭环。"""
    user_message = {
        "role": "user",
        "content": "请查询 002361 现在行情怎么样，并基于查询结果简要说明，不构成投资建议。",
    }
    messages: list[dict[str, Any]] = [user_message]

    # 第一次调用 DeepSeek：
    # 让模型返回 tool_calls。
    first_response = call_deepseek_chat(
        messages=messages,
        tools=build_tools_schema(),
        force_tool_name="get_realtime_quote",
    )
    assistant_message = extract_assistant_message(first_response)
    tool_calls = assistant_message.get("tool_calls") or []

    if not tool_calls:
        return f"""DeepSeek 没有返回 tool_calls。

第一次返回的 assistant message：
{json.dumps(assistant_message, ensure_ascii=False, indent=2)}
"""

    # 把 assistant 的 tool_calls 消息加入 messages。
    messages.append(assistant_message)

    # Python 程序真正执行工具。
    tool_results = [execute_tool_call(tool_call) for tool_call in tool_calls]

    # 把工具结果作为 tool 消息加入 messages。
    for result in tool_results:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            }
        )

    # 第二次调用 DeepSeek：
    # 让模型读取工具结果并生成最终回答。
    second_response = call_deepseek_chat(messages=messages)
    final_message = extract_assistant_message(second_response)

    return f"""======== 第一次 DeepSeek 返回的 tool_calls ========
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

======== Python 执行工具后的结果 ========
{json.dumps([result.__dict__ for result in tool_results], ensure_ascii=False, indent=2)}

======== 第二次发回 DeepSeek 的 messages ========
{json.dumps(messages, ensure_ascii=False, indent=2)}

======== DeepSeek 最终回答 ========
{final_message.get("content")}
"""


if __name__ == "__main__":
    try:
        print(run_deepseek_tool_calling_demo())
    except Exception as error:
        print(f"运行失败：{error}")
        sys.exit(1)
