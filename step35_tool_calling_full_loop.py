"""第 35 步：看懂 Tool Calling 的完整闭环。

第 34 步讲清楚了：

app.invoke({"messages": messages})
  ↓
ToolNode 执行工具
  ↓
messages 里新增 ToolMessage

但真实 Tool Calling 通常还差最后一步：

ToolMessage 工具结果
  ↓
再交给模型
  ↓
模型生成最终回答

完整闭环是：

HumanMessage：用户问题
  ↓
AIMessage：模型请求调用工具
  ↓
ToolNode：执行工具，生成 ToolMessage
  ↓
AIMessage：模型读取 ToolMessage，生成最终回答

本文件用单文件演示这个完整过程。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 用普通 Python 函数模拟模型两次输出。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class FullLoopState(TypedDict):
    """LangGraph 图状态。

    messages 是完整对话消息列表。

    add_messages 表示：
    每个节点返回新消息时，追加到原 messages 后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_realtime_quote(symbol: str) -> str:
    """查询股票实时行情。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def model_decide_tool_node(state: FullLoopState) -> dict:
    """第一次模型节点：决定是否调用工具。

    真实系统里：
    这里会调用大模型。

    大模型看到用户问：
    “002361 现在行情怎么样？”

    它不会直接回答，
    而是返回一个带 tool_calls 的 AIMessage。

    教学版里：
    我们手动构造这个 AIMessage。
    """
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
    return {"messages": [ai_message]}


def model_final_answer_node(state: FullLoopState) -> dict:
    """第二次模型节点：读取工具结果，生成最终回答。

    ToolNode 执行后，messages 最后一条通常是 ToolMessage。

    真实系统里：
    这一步会再次调用大模型，
    把完整 messages 发给模型，
    模型读取 ToolMessage 后生成自然语言回答。

    教学版里：
    我们用 Python 代码模拟最终回答。
    """
    last_message = state["messages"][-1]

    if not isinstance(last_message, ToolMessage):
        final_text = "没有拿到工具结果，无法生成最终回答。"
    else:
        final_text = (
            "根据实时行情工具返回的结果："
            f"{last_message.content} "
            "当前只是小幅上涨或反弹，需要结合技术面、新闻面和风控进一步判断。"
            "以上不构成投资建议。"
        )

    return {"messages": [AIMessage(content=final_text)]}


def build_app():
    """构建完整 Tool Calling 闭环图。

    图结构：

    START
      ↓
    model_decide_tool
      ↓
    tools
      ↓
    model_final_answer
      ↓
    END

    注意：
    tools 节点就是 LangGraph 自带 ToolNode。
    """
    graph = StateGraph(FullLoopState)

    graph.add_node("model_decide_tool", model_decide_tool_node)
    graph.add_node("tools", ToolNode([get_realtime_quote]))
    graph.add_node("model_final_answer", model_final_answer_node)

    graph.add_edge(START, "model_decide_tool")
    graph.add_edge("model_decide_tool", "tools")
    graph.add_edge("tools", "model_final_answer")
    graph.add_edge("model_final_answer", END)

    return graph.compile()


def describe_message(message: BaseMessage) -> str:
    """把一条消息转成便于阅读的文本。"""
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


def render_messages(messages: list[BaseMessage]) -> str:
    """渲染完整 messages。"""
    blocks: list[str] = []
    for index, message in enumerate(messages, start=1):
        blocks.append(f"--- 第 {index} 条消息 ---")
        blocks.append(describe_message(message))
        blocks.append("")
    return "\n".join(blocks)


def run_full_loop_demo() -> str:
    """运行完整 Tool Calling 闭环演示。"""
    app = build_app()
    initial_messages = [
        HumanMessage(content="002361 现在行情怎么样？"),
    ]

    output = app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": 10},
    )

    return f"""======== 初始输入 messages ========
{render_messages(initial_messages)}

======== 完整闭环结束后的 output["messages"] ========
{render_messages(output["messages"])}
"""


if __name__ == "__main__":
    print(run_full_loop_demo())
