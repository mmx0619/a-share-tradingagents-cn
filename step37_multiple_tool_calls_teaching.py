"""第 37 步：看懂一次模型输出多个 tool_calls。

第 36 步演示的是：

用户问行情
  ↓
模型生成 1 个 tool_call
  ↓
ToolNode 执行 1 个工具
  ↓
返回 1 个 ToolMessage

但真实场景里，模型一次可以请求多个工具。

例如用户问：

    002361 现在行情怎么样？最近有什么新闻？

模型可能一次返回两个 tool_calls：

1. get_realtime_quote
2. get_stock_news

ToolNode 会读取这两个 tool_calls，
分别执行对应工具，
然后返回两个 ToolMessage。

本文件专门演示：

AIMessage(tool_calls=[工具调用1, 工具调用2])
  ↓
ToolNode
  ↓
ToolMessage(行情结果)
ToolMessage(新闻结果)
  ↓
模型读取多个工具结果，生成最终回答

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 用 LangGraph 自带 ToolNode。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class MultiToolState(TypedDict):
    """图状态。

    messages 保存完整消息列表。
    add_messages 表示节点返回的新消息会追加到旧消息后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_realtime_quote(symbol: str) -> str:
    """查询股票实时行情。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


@tool
def get_stock_news(symbol: str, max_items: int = 3) -> str:
    """查询股票新闻。"""
    return (
        f"{symbol} 模拟新闻：最近 {max_items} 条事件包括龙虎榜、"
        "高换手、短线资金博弈。"
    )


def model_decide_tools_node(state: MultiToolState) -> dict:
    """第一次模型节点：一次生成多个工具调用。

    真实系统里，这一步由大模型完成。

    教学版里，我们手动构造 AIMessage。

    重点：

    tool_calls 是一个列表。
    列表里可以有多个工具调用。
    """
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_realtime_quote",
                "args": {"symbol": "002361"},
                "id": "call_quote_1",
                "type": "tool_call",
            },
            {
                "name": "get_stock_news",
                "args": {"symbol": "002361", "max_items": 3},
                "id": "call_news_1",
                "type": "tool_call",
            },
        ],
    )
    return {"messages": [ai_message]}


def model_final_answer_node(state: MultiToolState) -> dict:
    """第二次模型节点：读取多个 ToolMessage，生成最终回答。

    ToolNode 执行多个工具后，
    messages 里会追加多个 ToolMessage。

    这里我们把所有 ToolMessage 收集出来，
    模拟模型综合多个工具结果。
    """
    tool_messages = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    ]

    quote_text = "没有行情工具结果。"
    news_text = "没有新闻工具结果。"

    for message in tool_messages:
        if message.name == "get_realtime_quote":
            quote_text = str(message.content)
        elif message.name == "get_stock_news":
            news_text = str(message.content)

    final_text = f"""我已经读取到两个工具结果：

行情工具结果：
{quote_text}

新闻工具结果：
{news_text}

综合说明：
当前只是教学版模拟结果。真实项目里，模型会结合行情、新闻、技术面和风控再生成报告。
以上不构成投资建议。"""

    return {"messages": [AIMessage(content=final_text)]}


def build_app():
    """构建多工具调用图。

    图结构：

    START
      ↓
    model_decide_tools
      ↓
    tools
      ↓
    model_final_answer
      ↓
    END
    """
    graph = StateGraph(MultiToolState)

    graph.add_node("model_decide_tools", model_decide_tools_node)
    graph.add_node("tools", ToolNode([get_realtime_quote, get_stock_news]))
    graph.add_node("model_final_answer", model_final_answer_node)

    graph.add_edge(START, "model_decide_tools")
    graph.add_edge("model_decide_tools", "tools")
    graph.add_edge("tools", "model_final_answer")
    graph.add_edge("model_final_answer", END)

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

    return "\n".join(lines)


def render_messages(messages: list[BaseMessage]) -> str:
    """渲染消息列表。"""
    blocks: list[str] = []
    for index, message in enumerate(messages, start=1):
        blocks.append(f"--- 第 {index} 条消息 ---")
        blocks.append(describe_message(message))
        blocks.append("")
    return "\n".join(blocks)


def run_multi_tool_demo() -> str:
    """运行多工具调用演示。"""
    app = build_app()
    output = app.invoke(
        {
            "messages": [
                HumanMessage(content="002361 现在行情怎么样？最近有什么新闻？")
            ]
        },
        config={"recursion_limit": 10},
    )

    return f"""======== 多工具调用结束后的 messages ========
{render_messages(output["messages"])}
"""


if __name__ == "__main__":
    print(run_multi_tool_demo())
