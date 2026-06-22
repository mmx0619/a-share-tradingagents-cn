"""第 38 步：看懂 ToolNode 的工具错误处理。

第 37 步讲的是：

AIMessage 里可以有多个 tool_calls。
ToolNode 会执行多个工具，并返回多个 ToolMessage。

但真实工程里还会遇到另一个问题：

工具调用可能失败。

例如：

1. 模型传错参数。
2. 工具内部主动抛异常。
3. 网络请求失败。
4. 数据源暂时不可用。

那 ToolNode 会怎么办？

本文件专门演示：

AIMessage 请求调用一个会失败的工具
  ↓
ToolNode 执行工具
  ↓
工具抛出异常
  ↓
ToolNode 捕获异常
  ↓
ToolNode 返回一个 ToolMessage
  ↓
ToolMessage 里包含错误信息

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 使用 LangGraph 自带 ToolNode。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class ToolErrorState(TypedDict):
    """图状态。

    messages 保存完整消息列表。
    add_messages 表示节点返回的新消息会追加到旧消息后面。
    """

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def unstable_realtime_quote(symbol: str) -> str:
    """一个故意会失败的实时行情工具。

    真实项目里，失败可能来自：

    - 东方财富接口不可用。
    - 新浪接口超时。
    - 股票代码不存在。
    - 网络断开。

    教学版里，我们直接 raise RuntimeError，
    模拟工具内部出错。
    """
    raise RuntimeError(f"{symbol} 实时行情接口暂时不可用。")


def build_input_messages() -> list[BaseMessage]:
    """构造输入消息。

    这里手动构造 AIMessage.tool_calls，
    模拟模型请求调用 unstable_realtime_quote。
    """
    return [
        HumanMessage(content="002361 现在行情怎么样？"),
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
        ),
    ]


def build_app_with_builtin_error_handler():
    """构建启用内置错误捕获的 ToolNode 图。

    当前 LangGraph 版本里，
    ToolNode 默认不会吞掉所有工具异常。

    如果工具抛出异常，
    默认情况下图可能会直接失败。

    所以如果希望 ToolNode 把异常变成 ToolMessage，
    需要显式设置：

    handle_tool_errors=True

    工具抛异常后，
    ToolNode 不会直接让整个图崩掉，
    而是返回一个 ToolMessage，
    告诉后续模型“工具调用失败了”。
    """
    graph = StateGraph(ToolErrorState)
    graph.add_node(
        "tools",
        ToolNode(
            [unstable_realtime_quote],
            handle_tool_errors=True,
        ),
    )
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def build_app_with_custom_error_message():
    """构建使用自定义错误消息的 ToolNode 图。

    handle_tool_errors 可以传字符串。

    这样工具失败时，
    ToolNode 会把这个字符串放进 ToolMessage。

    真实项目里，你可以把它写成更适合模型继续处理的话。
    """
    graph = StateGraph(ToolErrorState)
    graph.add_node(
        "tools",
        ToolNode(
            [unstable_realtime_quote],
            handle_tool_errors="工具执行失败，进入保守兜底。请不要基于缺失数据给出交易结论。",
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


def run_case(title: str, app) -> str:
    """运行一个错误处理案例。"""
    input_messages = build_input_messages()
    output = app.invoke(
        {"messages": input_messages},
        config={"recursion_limit": 10},
    )
    new_messages = output["messages"][len(input_messages) :]

    return f"""======== {title} ========

输入消息：
{render_messages(input_messages)}

ToolNode 新增消息：
{render_messages(new_messages)}

完整 output["messages"]：
{render_messages(output["messages"])}
"""


def demo_toolnode_error_handling() -> None:
    """演示 ToolNode 内置错误处理和自定义错误处理。"""
    print(
        run_case(
            "情况 1：handle_tool_errors=True",
            build_app_with_builtin_error_handler(),
        )
    )
    print(
        run_case(
            "情况 2：自定义错误消息",
            build_app_with_custom_error_message(),
        )
    )


if __name__ == "__main__":
    demo_toolnode_error_handling()
