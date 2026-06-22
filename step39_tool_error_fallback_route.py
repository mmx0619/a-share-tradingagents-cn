"""第 39 步：工具失败后，如何路由到 fallback 节点。

第 38 步讲了：

工具函数报错
  ↓
ToolNode 捕获错误
  ↓
返回 ToolMessage
  ↓
ToolMessage.status = "error"

这一文件继续往下讲：

如果 ToolMessage 是 error，
程序如何不再进入正常回答节点，
而是进入 fallback_node？

核心流程：

HumanMessage
  ↓
AIMessage(tool_calls)
  ↓
ToolNode
  ↓
ToolMessage(status="success" 或 "error")
  ↓
route_after_tools
  ├─ success -> answer_node
  └─ error   -> fallback_node

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 使用 LangGraph 自带 ToolNode。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class ToolErrorFallbackState(TypedDict):
    """图状态。

    messages 保存完整消息列表。
    add_messages 表示新消息会追加到旧消息后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def stable_realtime_quote(symbol: str) -> str:
    """稳定的模拟行情工具。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


@tool
def unstable_realtime_quote(symbol: str) -> str:
    """故意失败的模拟行情工具。"""
    raise RuntimeError(f"{symbol} 实时行情接口暂时不可用。")


def model_success_tool_call_node(state: ToolErrorFallbackState) -> dict:
    """模拟模型请求调用稳定工具。"""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "stable_realtime_quote",
                        "args": {"symbol": "002361"},
                        "id": "call_stable_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def model_error_tool_call_node(state: ToolErrorFallbackState) -> dict:
    """模拟模型请求调用会失败的工具。"""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "unstable_realtime_quote",
                        "args": {"symbol": "002361"},
                        "id": "call_unstable_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def latest_tool_messages(state: ToolErrorFallbackState) -> list[ToolMessage]:
    """取出 messages 里的所有 ToolMessage。"""
    return [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    ]


def has_tool_error(state: ToolErrorFallbackState) -> bool:
    """判断是否有工具错误。

    ToolNode 返回的 ToolMessage 可能有 status 字段。

    如果 status == "error"，
    就说明工具执行失败。
    """
    for message in latest_tool_messages(state):
        status = getattr(message, "status", None)
        if status == "error":
            return True
    return False


def route_after_tools(state: ToolErrorFallbackState) -> str:
    """ToolNode 之后的路由函数。

    这是本文件最重要的函数。

    如果工具返回 error：
    - 不进入正常回答节点。
    - 进入 fallback_node。

    如果工具没有 error：
    - 进入 answer_node。
    """
    if has_tool_error(state):
        return "fallback"
    return "answer"


def answer_node(state: ToolErrorFallbackState) -> dict:
    """正常回答节点。

    只有工具成功时才进入这里。
    """
    tool_messages = latest_tool_messages(state)
    tool_text = "\n".join(str(message.content) for message in tool_messages)
    return {
        "messages": [
            AIMessage(
                content=(
                    "正常回答：工具调用成功。"
                    f"工具结果如下：{tool_text} "
                    "以上只是数据查询结果，不构成投资建议。"
                )
            )
        ]
    }


def fallback_node(state: ToolErrorFallbackState) -> dict:
    """兜底节点。

    只有工具失败时才进入这里。

    真实项目里，这里可以做：

    - 换备用数据源。
    - 降级为历史数据。
    - 提醒用户当前实时数据不可用。
    - 禁止基于缺失数据生成交易建议。
    """
    error_messages = [
        str(message.content)
        for message in latest_tool_messages(state)
        if getattr(message, "status", None) == "error"
    ]
    error_text = "\n".join(error_messages)
    return {
        "messages": [
            AIMessage(
                content=(
                    "兜底回答：工具执行失败，系统进入保守模式。"
                    "当前不能基于缺失数据给出交易结论。"
                    f"错误信息：{error_text}"
                )
            )
        ]
    }


def build_app(model_node_name: str):
    """构建工具错误路由图。

    参数 model_node_name 用来选择：

    - success：模型请求稳定工具。
    - error：模型请求失败工具。

    图结构：

    START
      ↓
    model
      ↓
    tools
      ↓
    route_after_tools
      ├─ answer
      └─ fallback
      ↓
    END
    """
    graph = StateGraph(ToolErrorFallbackState)

    if model_node_name == "success":
        graph.add_node("model", model_success_tool_call_node)
    elif model_node_name == "error":
        graph.add_node("model", model_error_tool_call_node)
    else:
        raise ValueError(f"未知 model_node_name：{model_node_name}")

    graph.add_node(
        "tools",
        ToolNode(
            [stable_realtime_quote, unstable_realtime_quote],
            handle_tool_errors=True,
        ),
    )
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "model")
    graph.add_edge("model", "tools")
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        path_map={
            "answer": "answer",
            "fallback": "fallback",
        },
    )
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

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


def run_case(title: str, model_node_name: str, question: str) -> str:
    """运行一个案例。"""
    app = build_app(model_node_name)
    output = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 10},
    )
    return f"""======== {title} ========

最终 messages：
{render_messages(output["messages"])}
"""


def demo_tool_error_fallback_route() -> None:
    """演示工具成功和工具失败两种路由。"""
    print(
        run_case(
            title="情况 1：工具成功，进入 answer_node",
            model_node_name="success",
            question="002361 现在行情怎么样？",
        )
    )
    print(
        run_case(
            title="情况 2：工具失败，进入 fallback_node",
            model_node_name="error",
            question="002361 现在行情怎么样？",
        )
    )


if __name__ == "__main__":
    demo_tool_error_fallback_route()
