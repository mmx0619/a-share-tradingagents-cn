"""运行正式 AKShare ToolNode 最小示例。

这是正式工程目录里的第一个可运行示例。

它复用：

tradingagents_cn.tools.akshare_tools.get_akshare_tools()

流程：

HumanMessage
  ↓
AIMessage(tool_calls)
  ↓
ToolNode(get_akshare_tools())
  ↓
ToolMessage

注意：
这个示例会访问 AKShare 背后的公开行情源。
如果没有网络权限，ToolNode 会返回错误 ToolMessage。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# 允许用户直接运行本文件：
#
# python tradingagents_cn/examples/run_akshare_toolnode_demo.py
#
# 直接运行子目录里的文件时，
# Python 默认不一定能找到项目根目录。
#
# 所以这里把项目根目录加入 sys.path，
# 让 from tradingagents_cn... 可以正常导入。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents_cn.tools.akshare_tools import get_akshare_tools


class AkshareToolNodeDemoState(TypedDict):
    """示例图状态。

    messages 是 LangGraph / ToolNode 使用的消息列表。
    """

    messages: Annotated[list[BaseMessage], add_messages]


def build_input_messages(symbol: str) -> list[BaseMessage]:
    """构造输入消息。

    真实项目中，AIMessage 应该由大模型生成。
    当前示例中，为了专注验证 ToolNode 和 AKShare 工具，
    先手动构造 tool_calls。
    """
    return [
        HumanMessage(content=f"{symbol} 现在行情怎么样？"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "akshare_realtime_quote",
                    "args": {"symbol": symbol},
                    "id": "call_akshare_quote_1",
                    "type": "tool_call",
                }
            ],
        ),
    ]


def build_app():
    """构建最小 ToolNode 图。"""
    graph = StateGraph(AkshareToolNodeDemoState)
    graph.add_node(
        "tools",
        ToolNode(
            get_akshare_tools(),
            handle_tool_errors=(
                "AKShare 工具执行失败，无法获取行情。"
                "请不要基于缺失数据给出交易结论。"
            ),
        ),
    )
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def describe_message(message: BaseMessage) -> str:
    """把消息转成便于阅读的文本。"""
    lines = [
        f"类型：{message.__class__.__name__}",
        f"内容：{message.content}",
    ]

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        lines.append(f"tool_calls：{tool_calls}")

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        lines.append(f"tool_call_id：{tool_call_id}")

    name = getattr(message, "name", None)
    if name:
        lines.append(f"name：{name}")

    status = getattr(message, "status", None)
    if status:
        lines.append(f"status：{status}")

    return "\n".join(lines)


def render_messages(messages: list[BaseMessage]) -> str:
    """渲染消息列表。"""
    blocks: list[str] = []
    for index, message in enumerate(messages, start=1):
        blocks.append(f"--- 第 {index} 条消息 ---")
        blocks.append(describe_message(message))
        blocks.append("")
    return "\n".join(blocks)


def run_demo(symbol: str = "002361") -> str:
    """运行正式 AKShare ToolNode 最小示例。"""
    app = build_app()
    input_messages = build_input_messages(symbol)
    output = app.invoke(
        {"messages": input_messages},
        config={"recursion_limit": 10},
    )
    new_messages = output["messages"][len(input_messages) :]

    return f"""======== 输入消息 ========
{render_messages(input_messages)}

======== ToolNode 新增消息 ========
{render_messages(new_messages)}

======== 完整输出消息 ========
{render_messages(output["messages"])}
"""


if __name__ == "__main__":
    print(run_demo())
