"""第 33 步：使用 LangGraph 自带的 ToolNode。

第 32 步我们自己写了一个教学版 tool_node。

它做的事情是：

读取 tool_call
  ↓
找到 Python 工具函数
  ↓
执行工具函数
  ↓
写入 tool_result

这一文件开始使用 LangGraph 自带的 ToolNode：

from langgraph.prebuilt import ToolNode

你可以把它理解成：

LangGraph 已经帮我们写好了一个通用工具执行节点。

它要求我们做两件事：

1. 用 @tool 把普通 Python 函数声明成“工具”。
2. 给 ToolNode 一个带 tool_calls 的 AIMessage。

ToolNode 会自动：

- 读取 AIMessage.tool_calls。
- 根据 tool_calls 里的 name 找工具。
- 把 args 传给工具函数。
- 执行工具。
- 返回 ToolMessage。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 只演示真正的 ToolNode 如何执行工具。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class ToolNodeGraphState(TypedDict):
    """ToolNode 所在 LangGraph 图的状态。

    这里只需要一个字段：

    messages

    messages 是消息列表。

    Annotated[list[BaseMessage], add_messages] 的意思是：

    每个节点返回新的 messages 时，
    LangGraph 不要直接覆盖旧消息，
    而是把新消息追加到旧消息后面。

    这很适合 ToolNode：

    输入消息：
    HumanMessage + AIMessage(tool_calls)

    ToolNode 输出：
    ToolMessage

    最终 messages：
    HumanMessage + AIMessage(tool_calls) + ToolMessage
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_realtime_quote(symbol: str) -> str:
    """查询股票实时行情。

    教学版不联网，只返回模拟行情。

    真实项目里，这个工具可以替换成：
    - AKShare 实时行情
    - 东方财富实时行情
    - 新浪实时行情
    """
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


@tool
def get_stock_news(symbol: str, max_items: int = 3) -> str:
    """查询股票新闻。

    教学版不联网，只返回模拟新闻。

    真实项目里，这个工具可以替换成：
    - 东方财富个股新闻
    - 财联社新闻
    - 巨潮资讯公告
    """
    return (
        f"{symbol} 模拟新闻：最近 {max_items} 条事件包括龙虎榜、"
        "高换手、短线资金博弈。"
    )


def build_tool_node() -> ToolNode:
    """创建真正的 LangGraph ToolNode。

    ToolNode 接收一个工具列表。

    这些工具必须是用 @tool 包装过的函数，
    或者是 LangChain/LangGraph 认可的工具对象。
    """
    return ToolNode(
        [
            get_realtime_quote,
            get_stock_news,
        ]
    )


def build_ai_message_with_quote_tool_call(symbol: str) -> AIMessage:
    """构造一个“模型想调用行情工具”的 AIMessage。

    真实系统里，这个 AIMessage 会来自大模型。

    教学版里不调真实模型，
    所以我们手动构造一个 AIMessage。

    重点看 tool_calls：

    - name：工具名，必须和 @tool 函数名一致。
    - args：工具参数。
    - id：本次工具调用 ID。
    - type：固定写 tool_call。
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_realtime_quote",
                "args": {"symbol": symbol},
                "id": "call_quote_1",
                "type": "tool_call",
            }
        ],
    )


def build_ai_message_with_news_tool_call(symbol: str) -> AIMessage:
    """构造一个“模型想调用新闻工具”的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_stock_news",
                "args": {"symbol": symbol, "max_items": 3},
                "id": "call_news_1",
                "type": "tool_call",
            }
        ],
    )


def run_toolnode_once(messages: list[BaseMessage]) -> list[BaseMessage]:
    """运行一次 ToolNode。

    当前版本的 LangGraph 更推荐把 ToolNode 放进图里运行。

    所以这里创建一个最小图：

    START
      ↓
    tools
      ↓
    END

    tools 节点就是 LangGraph 自带的 ToolNode。

    图的输入是：

    {
        "messages": [...]
    }

    其中最后一条消息通常是 AIMessage，
    并且这个 AIMessage 里带有 tool_calls。

    图运行结束后，messages 里会追加 ToolMessage。

    返回值只返回新追加的 ToolMessage，
    方便你看清楚 ToolNode 的输出。
    """
    graph = StateGraph(ToolNodeGraphState)
    graph.add_node("tools", build_tool_node())
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    app = graph.compile()

    output = app.invoke({"messages": messages})
    all_messages = output["messages"]
    return all_messages[len(messages) :]


def render_messages(messages: list[BaseMessage]) -> str:
    """把消息列表渲染成便于阅读的文本。"""
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        lines.append(f"消息 {index}")
        lines.append(f"类型：{message.__class__.__name__}")
        lines.append(f"内容：{message.content}")

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            lines.append(f"工具调用：{tool_calls}")

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            lines.append(f"工具调用 ID：{tool_call_id}")

        name = getattr(message, "name", None)
        if name:
            lines.append(f"工具名称：{name}")

        lines.append("")

    return "\n".join(lines)


def demo_quote_toolnode() -> str:
    """演示 ToolNode 执行行情工具。"""
    human_message = HumanMessage(content="002361 现在行情怎么样？")
    ai_message = build_ai_message_with_quote_tool_call("002361")

    input_messages = [human_message, ai_message]
    tool_messages = run_toolnode_once(input_messages)

    return f"""======== 输入给 ToolNode 的消息 ========
{render_messages(input_messages)}

======== ToolNode 返回的工具消息 ========
{render_messages(tool_messages)}
"""


def demo_news_toolnode() -> str:
    """演示 ToolNode 执行新闻工具。"""
    human_message = HumanMessage(content="002361 最近有什么新闻？")
    ai_message = build_ai_message_with_news_tool_call("002361")

    input_messages = [human_message, ai_message]
    tool_messages = run_toolnode_once(input_messages)

    return f"""======== 输入给 ToolNode 的消息 ========
{render_messages(input_messages)}

======== ToolNode 返回的工具消息 ========
{render_messages(tool_messages)}
"""


def demo_real_toolnode() -> None:
    """运行两个 ToolNode 演示。"""
    print("======== 情况 1：ToolNode 执行 get_realtime_quote ========")
    print(demo_quote_toolnode())

    print("======== 情况 2：ToolNode 执行 get_stock_news ========")
    print(demo_news_toolnode())


if __name__ == "__main__":
    demo_real_toolnode()


    
