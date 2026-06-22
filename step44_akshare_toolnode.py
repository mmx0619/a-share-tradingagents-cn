"""第 44 步：真实 AKShare ToolNode。

前面第 43 步已经跑通了真实 DeepSeek Tool Calling。

但第 43 步里的工具还是模拟工具：

get_realtime_quote(symbol)
  -> 返回固定模拟行情

这一文件做最后一个关键知识点：

把 ToolNode 里的工具换成真实 AKShare 数据工具。

流程：

HumanMessage：用户问 002361 现在行情怎么样
  ↓
AIMessage：模拟模型请求调用 akshare_realtime_quote
  ↓
ToolNode：执行真实 AKShare 工具
  ↓
ToolMessage：返回真实行情文本

注意：

1. 这一步不调用 DeepSeek。
2. 这一步会联网请求 AKShare 背后的公开数据源。
3. 这一步只验证“真实 AKShare 工具能被 ToolNode 执行”。
4. 真正完整版本是：DeepSeek 返回 tool_calls，ToolNode 执行 AKShare 工具，再把 ToolMessage 发回 DeepSeek。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import step19_realtime_quote as realtime_quote


class AkshareToolState(TypedDict):
    """LangGraph 图状态。

    messages 保存完整消息列表。

    add_messages 表示：
    ToolNode 返回的 ToolMessage 会追加到原消息列表后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def akshare_realtime_quote(symbol: str) -> str:
    """查询 A 股实时/近实时行情快照。

    参数：
    - symbol：6 位 A 股股票代码，例如 002361。

    这个工具内部复用第 19 步：

    step19_realtime_quote.get_realtime_quote()

    第 19 步会：

    1. 优先尝试东方财富实时行情。
    2. 如果失败，切换到新浪备用源。
    3. 最后统一渲染成文本。
    """
    quote = realtime_quote.get_realtime_quote(symbol)
    return realtime_quote.render_realtime_quote_text(quote)


def build_input_messages(symbol: str) -> list[BaseMessage]:
    """构造输入 ToolNode 的消息。

    真实系统里：
    AIMessage 应该由 DeepSeek 生成。

    教学版里：
    我们手动构造 AIMessage.tool_calls，
    让 ToolNode 专注执行真实 AKShare 工具。
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
    """构建只包含一个真实 AKShare ToolNode 的图。

    图结构：

    START
      ↓
    tools
      ↓
    END
    """
    graph = StateGraph(AkshareToolState)
    graph.add_node(
        "tools",
        ToolNode(
            [akshare_realtime_quote],
            handle_tool_errors=(
                "AKShare 行情工具执行失败，无法获取实时行情。"
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


def run_akshare_toolnode_demo(symbol: str = "002361") -> str:
    """运行真实 AKShare ToolNode 演示。"""
    app = build_app()
    input_messages = build_input_messages(symbol)
    output = app.invoke(
        {"messages": input_messages},
        config={"recursion_limit": 10},
    )
    new_messages = output["messages"][len(input_messages) :]

    return f"""======== 输入 ToolNode 前的 messages ========
{render_messages(input_messages)}

======== ToolNode 新增的 ToolMessage ========
{render_messages(new_messages)}

======== 完整 output["messages"] ========
{render_messages(output["messages"])}
"""


if __name__ == "__main__":
    print(run_akshare_toolnode_demo())
