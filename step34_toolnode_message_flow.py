"""第 34 步：看懂 ToolNode 前后的 messages 变化。

第 33 步里最关键的一句是：

output = app.invoke({"messages": messages})

你问：

这一步的输出是什么？

本文件专门回答这个问题。

我们用一个很小的例子：

human_message = HumanMessage(content="002361 现在行情怎么样？")

然后手动构造一个 AIMessage：

AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_realtime_quote",
            "args": {"symbol": "002361"},
            "id": "call_quote_1",
            "type": "tool_call",
        }
    ],
)

输入给 LangGraph 的 messages 一开始有 2 条：

1. HumanMessage：用户问题
2. AIMessage：模型请求调用工具

ToolNode 执行后，output["messages"] 会变成 3 条：

1. HumanMessage：用户问题
2. AIMessage：模型请求调用工具
3. ToolMessage：工具执行结果

所以：

output = app.invoke({"messages": messages})

返回的是一个字典。

其中：

output["messages"]

就是更新后的完整消息列表。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class MessageFlowState(TypedDict):
    """LangGraph 图状态。

    这里只保留 messages 一个字段。

    add_messages 的作用是：
    节点返回新的消息时，不覆盖旧消息，而是追加到旧消息后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_realtime_quote(symbol: str) -> str:
    """查询股票实时行情。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def build_input_messages() -> list[BaseMessage]:
    """构造传入 app.invoke 的原始 messages。

    这里不调用真实大模型。

    我们手动构造：

    - HumanMessage：用户问行情。
    - AIMessage：模型请求调用 get_realtime_quote。
    """
    human_message = HumanMessage(content="002361 现在行情怎么样？")
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_realtime_quote",
                "args": {"symbol": "002361"},
                "id": "call_quote_1",
                "type": "tool_call",
            }
        ],
    )
    return [human_message, ai_message]


def build_app():
    """构建只有一个 ToolNode 的 LangGraph。

    图结构：

    START
      ↓
    tools
      ↓
    END
    """
    graph = StateGraph(MessageFlowState)
    graph.add_node("tools", ToolNode([get_realtime_quote]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def describe_message(message: BaseMessage) -> str:
    """把单条消息转成更容易读的文本。"""
    lines = [
        f"消息类型：{message.__class__.__name__}",
        f"content：{message.content}",
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


def describe_messages(title: str, messages: list[BaseMessage]) -> str:
    """把消息列表转成文本。"""
    blocks = [title]
    for index, message in enumerate(messages, start=1):
        blocks.append(f"\n--- 第 {index} 条消息 ---")
        blocks.append(describe_message(message))
    return "\n".join(blocks)


def run_message_flow_demo() -> str:
    """运行 messages 输入输出演示。"""
    app = build_app()
    input_messages = build_input_messages()

    # 这就是你问的关键语句。
    #
    # 输入：
    # {
    #     "messages": [HumanMessage, AIMessage]
    # }
    #
    # 输出：
    # {
    #     "messages": [HumanMessage, AIMessage, ToolMessage]
    # }
    output = app.invoke({"messages": input_messages})

    output_messages = output["messages"]

    # 新增消息就是 output_messages 去掉原来 input_messages 的部分。
    #
    # 原来有 2 条：
    # HumanMessage、AIMessage
    #
    # 运行后有 3 条：
    # HumanMessage、AIMessage、ToolMessage
    #
    # 所以新增的就是最后 1 条 ToolMessage。
    new_messages = output_messages[len(input_messages) :]

    return f"""{describe_messages("======== 输入 app.invoke 之前的 messages ========", input_messages)}

{describe_messages("======== app.invoke 返回后的 output['messages'] ========", output_messages)}

{describe_messages("======== 本次新增的 messages ========", new_messages)}
"""


if __name__ == "__main__":
    print(run_message_flow_demo())
