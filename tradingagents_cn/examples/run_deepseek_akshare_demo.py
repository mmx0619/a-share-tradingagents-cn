"""运行 DeepSeek + AKShare 工具的最小正式闭环。

这是第 48 步。

它把两个已经验证过的能力合起来：

1. 第 43 步：真实 DeepSeek Tool Calling。
2. 第 47 步：正式 AKShare ToolNode 工具。

当前流程：

用户问题
  ↓
DeepSeek 返回 tool_calls
  ↓
Python 执行 tradingagents_cn.tools.akshare_tools 中的真实 AKShare 工具
  ↓
把工具结果作为 role="tool" 消息发回 DeepSeek
  ↓
DeepSeek 生成最终回答

注意：

1. 本文件会真实调用 DeepSeek API。
2. 本文件会真实访问 AKShare 背后的公开行情源。
3. API Key 只从 DEEPSEEK_API_KEY 环境变量读取，不写入代码。
4. 这是最小闭环，还没有接入新闻、技术指标、风控、交易员。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# 允许用户直接运行本文件：
#
# python tradingagents_cn/examples/run_deepseek_akshare_demo.py
#
# 直接运行子目录里的文件时，
# Python 默认不一定能找到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents_cn.tools.akshare_tools import get_akshare_tools


DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
TOOL_NAME = "akshare_realtime_quote"


@dataclass
class ToolExecutionResult:
    """工具执行结果。"""

    tool_call_id: str
    tool_name: str
    content: str


def get_env(name: str) -> str | None:
    """读取环境变量。"""
    value = os.environ.get(name)
    if value:
        return value
    return None


def build_tools_schema() -> list[dict[str, Any]]:
    """构造发给 DeepSeek 的 tools schema。

    注意：
    这里的工具名必须和正式工具名一致：

    akshare_realtime_quote

    否则 DeepSeek 返回的 tool_call 无法映射到 Python 工具。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "查询 A 股股票的实时/近实时行情快照。"
                    "数据来自 AKShare 背后的公开行情源。"
                ),
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
    """调用 DeepSeek Chat Completions API。"""
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
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def extract_assistant_message(response_json: dict[str, Any]) -> dict[str, Any]:
    """从 DeepSeek 返回结果中取出 assistant message。"""
    choices = response_json.get("choices")
    if not choices:
        raise ValueError("DeepSeek 返回结果中没有 choices。")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("DeepSeek 返回结果中没有合法 message。")

    return message


def get_tool_map() -> dict[str, Any]:
    """把正式工具列表转换成工具名字典。

    get_akshare_tools() 返回的是 LangChain 工具对象列表。

    这里转换成：

    {
        "akshare_realtime_quote": 工具对象
    }

    方便根据 DeepSeek 返回的 tool_call.function.name 找到工具。
    """
    return {
        tool_item.name: tool_item
        for tool_item in get_akshare_tools()
    }


def execute_tool_call(tool_call: dict[str, Any]) -> ToolExecutionResult:
    """执行 DeepSeek 返回的一个 tool_call。"""
    tool_call_id = str(tool_call.get("id", ""))
    function_info = tool_call.get("function", {})
    tool_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", "{}")

    if not tool_call_id:
        raise ValueError("tool_call 缺少 id。")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_call 缺少 function.name。")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ValueError(f"工具参数不是合法 JSON：{raw_arguments}") from error

    tool_map = get_tool_map()
    tool_item = tool_map.get(tool_name)
    if tool_item is None:
        raise ValueError(f"不支持的工具名：{tool_name}")

    # LangChain 工具对象使用 invoke 执行。
    content = tool_item.invoke(arguments)
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=str(content),
    )


def run_deepseek_akshare_demo(symbol: str = "002361") -> str:
    """运行 DeepSeek + AKShare 最小正式闭环。"""
    user_message = {
        "role": "user",
        "content": (
            f"请查询 {symbol} 现在行情怎么样，"
            "并基于查询结果简要说明，本系统用于个人投资研究辅助，最终决策由使用者自行确认。"
        ),
    }
    messages: list[dict[str, Any]] = [user_message]

    first_response = call_deepseek_chat(
        messages=messages,
        tools=build_tools_schema(),
        force_tool_name=TOOL_NAME,
    )
    assistant_message = extract_assistant_message(first_response)
    tool_calls = assistant_message.get("tool_calls") or []

    if not tool_calls:
        return f"""DeepSeek 没有返回 tool_calls。

第一次返回的 assistant message：
{json.dumps(assistant_message, ensure_ascii=False, indent=2)}
"""

    messages.append(assistant_message)

    tool_results = [execute_tool_call(tool_call) for tool_call in tool_calls]
    for result in tool_results:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": result.content,
            }
        )

    second_response = call_deepseek_chat(messages=messages)
    final_message = extract_assistant_message(second_response)

    return f"""======== 第一次 DeepSeek 返回的 tool_calls ========
{json.dumps(tool_calls, ensure_ascii=False, indent=2)}

======== Python 执行正式 AKShare 工具后的结果 ========
{json.dumps([result.__dict__ for result in tool_results], ensure_ascii=False, indent=2)}

======== 发回 DeepSeek 的 messages ========
{json.dumps(messages, ensure_ascii=False, indent=2)}

======== DeepSeek 最终回答 ========
{final_message.get("content")}
"""


if __name__ == "__main__":
    print(run_deepseek_akshare_demo())
