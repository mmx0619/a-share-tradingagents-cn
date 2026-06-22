"""第 36 步：用条件路由实现真实一点的 Tool Calling 图。

第 35 步的图是固定顺序：

model_decide_tool
  ↓
tools
  ↓
model_final_answer

这有助于理解闭环，但真实 Agent 通常不是写死一定调用工具。

真实情况更像这样：

模型节点
  ↓
判断最后一条 AIMessage 有没有 tool_calls
  ↓
有 tool_calls：进入 ToolNode
  ↓
ToolNode 返回 ToolMessage
  ↓
再回到模型节点
  ↓
模型读取 ToolMessage，生成最终回答
  ↓
这次没有 tool_calls
  ↓
流程结束

这个文件重点讲：

1. 如何判断模型是否请求调用工具。
2. 如何用条件边决定去 ToolNode 还是结束。
3. ToolNode 执行完后，为什么要回到模型节点。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 只用 LangGraph、ToolNode、消息对象和模拟模型函数。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class ConditionalToolState(TypedDict):
    """图状态。

    messages 保存完整对话消息。

    add_messages 表示：
    节点返回的新消息会追加到旧消息后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_realtime_quote(symbol: str) -> str:
    """查询股票实时行情。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def model_node(state: ConditionalToolState) -> dict:
    """模拟模型节点。

    这个节点会被调用两次：

    第一次：
    - 最后一条消息是 HumanMessage。
    - 模型判断用户在问行情。
    - 返回带 tool_calls 的 AIMessage。

    第二次：
    - 最后一条消息是 ToolMessage。
    - 模型读取工具结果。
    - 返回最终回答 AIMessage。
    - 这次不带 tool_calls。
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage):
        question = last_message.content
        if "行情" in question or "现在" in question or "价格" in question:
            return {
                "messages": [
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
                ]
            }

        return {
            "messages": [
                AIMessage(content="这个问题暂时没有触发工具调用。")
            ]
        }

    if isinstance(last_message, ToolMessage):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "我已经读取到工具结果："
                        f"{last_message.content} "
                        "这只是行情查询结果，不构成投资建议。"
                    )
                )
            ]
        }

    return {
        "messages": [
            AIMessage(content="没有识别到可处理的消息类型。")
        ]
    }


def route_after_model(state: ConditionalToolState) -> str:
    """模型节点之后的条件路由。

    这是本文件最重要的函数。

    它读取最后一条消息。

    如果最后一条是 AIMessage，并且里面有 tool_calls，
    说明模型请求调用工具，于是返回 "tools"。

    如果没有 tool_calls，
    说明模型已经给出最终回答，于是返回 "end"。
    """
    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"

    return "end"


def build_app():
    """构建条件 Tool Calling 图。

    图结构：

    START
      ↓
    model
      ↓
    route_after_model
      ├─ 有 tool_calls -> tools -> model
      └─ 没有 tool_calls -> END
    """
    graph = StateGraph(ConditionalToolState)

    graph.add_node("model", model_node)
    graph.add_node("tools", ToolNode([get_realtime_quote]))

    graph.add_edge(START, "model")

    graph.add_conditional_edges(
        "model",
        route_after_model,
        path_map={
            "tools": "tools",
            "end": END,
        },
    )

    # 工具执行完以后，要回到模型节点。
    #
    # 因为工具只返回 ToolMessage，
    # 还需要模型读取 ToolMessage，
    # 再生成最终自然语言回答。
    graph.add_edge("tools", "model")

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


def run_case(question: str) -> str:
    """运行一个用户问题。"""
    app = build_app()
    output = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 10},
    )
    return f"""用户问题：
{question}

最终 messages：
{render_messages(output["messages"])}
"""


def demo_conditional_tool_calling() -> None:
    """演示触发工具和不触发工具两种情况。"""
    print("======== 情况 1：触发工具调用 ========")
    print(run_case("002361 现在行情怎么样？"))

    print("======== 情况 2：不触发工具调用 ========")
    print(run_case("你好，介绍一下这个系统。"))


if __name__ == "__main__":
    demo_conditional_tool_calling()
