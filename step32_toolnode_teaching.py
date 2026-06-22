"""第 32 步：单文件看懂 ToolNode。

第 31 步讲了 Tool Calling：

模型返回：
    我要调用 get_realtime_quote，参数是 {"symbol": "002361"}

程序执行：
    get_realtime_quote("002361")

这一文件继续讲：

ToolNode 是什么？

你可以先简单理解为：

ToolNode 就是 LangGraph 里的一个节点。

这个节点专门做一件事：

读取模型生成的工具调用请求
  ↓
找到对应 Python 工具函数
  ↓
执行工具函数
  ↓
把工具结果写回 state

所以 ToolNode 不是大模型。
ToolNode 是程序里的“工具执行节点”。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面的复杂模块。
- 只用 LangGraph 和普通 Python 函数。

本文件流程：

用户问题
  ↓
model_node：模拟模型决定要不要调用工具
  ↓
route_after_model：如果有工具调用，就进入 tool_node
  ↓
tool_node：真正执行工具
  ↓
answer_node：根据工具结果生成最终回答
  ↓
END
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


ToolName = Literal["get_realtime_quote", "get_stock_news"]


@dataclass
class ToolCall:
    """模型生成的工具调用请求。

    字段说明：

    - name：工具名。
    - args：工具参数。

    真实 Tool Calling 里，
    这部分通常由大模型 API 返回。
    """

    name: ToolName
    args: dict[str, Any]


@dataclass
class ToolResult:
    """ToolNode 执行工具后的结果。"""

    name: str
    success: bool
    content: str


class ToolNodeState(TypedDict):
    """LangGraph 共享状态。

    字段说明：

    - user_question：用户原始问题。
    - tool_call：模型决定调用的工具。
    - tool_result：工具执行结果。
    - final_answer：最终回答。
    - step_log：执行日志。
    """

    user_question: str
    tool_call: ToolCall | None
    tool_result: ToolResult | None
    final_answer: str
    step_log: list[str]


def create_initial_state(user_question: str) -> ToolNodeState:
    """创建初始状态。"""
    return {
        "user_question": user_question,
        "tool_call": None,
        "tool_result": None,
        "final_answer": "",
        "step_log": [],
    }


def append_log(state: ToolNodeState, message: str) -> list[str]:
    """追加执行日志。"""
    return [*state["step_log"], message]


def get_realtime_quote(symbol: str) -> str:
    """模拟实时行情工具。"""
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def get_stock_news(symbol: str) -> str:
    """模拟新闻工具。"""
    return f"{symbol} 模拟新闻：龙虎榜、高换手、短线资金博弈。"


def model_node(state: ToolNodeState) -> dict:
    """模拟大模型节点。

    真实系统里，这一步会调用大模型。

    大模型会判断：

    - 用户是不是在问行情？
    - 用户是不是在问新闻？
    - 是否需要调用工具？

    教学版里不用真实模型，
    只用 if 判断模拟模型选择工具。
    """
    question = state["user_question"]

    if "行情" in question or "价格" in question or "现在" in question:
        tool_call = ToolCall(
            name="get_realtime_quote",
            args={"symbol": "002361"},
        )
        return {
            "tool_call": tool_call,
            "step_log": append_log(
                state,
                "model_node：模型决定调用 get_realtime_quote",
            ),
        }

    if "新闻" in question or "消息" in question:
        tool_call = ToolCall(
            name="get_stock_news",
            args={"symbol": "002361"},
        )
        return {
            "tool_call": tool_call,
            "step_log": append_log(
                state,
                "model_node：模型决定调用 get_stock_news",
            ),
        }

    return {
        "final_answer": "模型判断：这个问题暂时不需要调用工具，无法给出数据查询结果。",
        "step_log": append_log(state, "model_node：模型没有选择工具"),
    }


def tool_node(state: ToolNodeState) -> dict:
    """工具执行节点，也就是教学版 ToolNode。

    这个节点不负责思考。

    它只负责执行：

    1. 读取 state["tool_call"]。
    2. 看 tool_call.name 是哪个工具。
    3. 取出参数。
    4. 调用对应 Python 函数。
    5. 把结果写入 state["tool_result"]。

    这就是 ToolNode 的核心作用。
    """
    tool_call = state["tool_call"]
    if tool_call is None:
        return {
            "tool_result": ToolResult(
                name="fallback",
                success=False,
                content="没有工具调用请求，无法执行工具。",
            ),
            "step_log": append_log(state, "tool_node：缺少 tool_call，进入工具兜底"),
        }

    symbol = tool_call.args.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        return {
            "tool_result": ToolResult(
                name=tool_call.name,
                success=False,
                content="工具参数错误：缺少合法 symbol。",
            ),
            "step_log": append_log(state, "tool_node：工具参数校验失败"),
        }

    if tool_call.name == "get_realtime_quote":
        content = get_realtime_quote(symbol)
        return {
            "tool_result": ToolResult(
                name=tool_call.name,
                success=True,
                content=content,
            ),
            "step_log": append_log(state, "tool_node：已执行 get_realtime_quote"),
        }

    if tool_call.name == "get_stock_news":
        content = get_stock_news(symbol)
        return {
            "tool_result": ToolResult(
                name=tool_call.name,
                success=True,
                content=content,
            ),
            "step_log": append_log(state, "tool_node：已执行 get_stock_news"),
        }

    return {
        "tool_result": ToolResult(
            name=str(tool_call.name),
            success=False,
            content=f"未知工具：{tool_call.name}",
        ),
        "step_log": append_log(state, "tool_node：未知工具，进入工具兜底"),
    }


def answer_node(state: ToolNodeState) -> dict:
    """最终回答节点。

    这个节点模拟：

    大模型拿到工具结果后，
    组织成用户能读懂的自然语言回答。
    """
    tool_result = state["tool_result"]
    if tool_result is None:
        answer = state["final_answer"] or "没有工具结果，无法生成回答。"
    elif tool_result.success:
        answer = (
            f"根据工具 {tool_result.name} 的查询结果：{tool_result.content} "
            "以上只是数据查询结果，不构成投资建议。"
        )
    else:
        answer = f"工具执行失败，进入兜底回答。原因：{tool_result.content}"

    return {
        "final_answer": answer,
        "step_log": append_log(state, "answer_node：生成最终回答"),
    }


def route_after_model(state: ToolNodeState) -> str:
    """模型节点之后的路由。

    如果模型生成了 tool_call，
    就进入 tool_node。

    如果模型没有生成 tool_call，
    就直接进入 answer_node。
    """
    if state["tool_call"] is None:
        return "answer_node"
    return "tool_node"


def build_graph():
    """构建 LangGraph。

    图结构：

    START
      ↓
    model_node
      ↓
    tool_node 或 answer_node
      ↓
    answer_node
      ↓
    END
    """
    graph = StateGraph(ToolNodeState)

    graph.add_node("model_node", model_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("answer_node", answer_node)

    graph.add_edge(START, "model_node")

    graph.add_conditional_edges(
        "model_node",
        route_after_model,
        path_map={
            "tool_node": "tool_node",
            "answer_node": "answer_node",
        },
    )

    graph.add_edge("tool_node", "answer_node")
    graph.add_edge("answer_node", END)

    return graph.compile()


def run_case(user_question: str) -> str:
    """运行一个 ToolNode 案例。"""
    app = build_graph()
    final_state = app.invoke(
        create_initial_state(user_question),
        config={"recursion_limit": 10},
    )
    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(final_state["step_log"], start=1)
    )
    tool_call = final_state["tool_call"]
    tool_result = final_state["tool_result"]

    tool_call_text = "无"
    if tool_call is not None:
        tool_call_text = f"name={tool_call.name}, args={tool_call.args}"

    tool_result_text = "无"
    if tool_result is not None:
        tool_result_text = (
            f"name={tool_result.name}, "
            f"success={tool_result.success}, "
            f"content={tool_result.content}"
        )

    return f"""用户问题：
{user_question}

执行日志：
{step_log}

模型生成的工具调用：
{tool_call_text}

ToolNode 执行结果：
{tool_result_text}

最终回答：
{final_state["final_answer"]}
"""


def demo_toolnode() -> None:
    """演示 ToolNode 的三种情况。"""
    print("======== 情况 1：用户问行情，ToolNode 执行行情工具 ========")
    print(run_case("002361 现在行情怎么样？"))

    print("======== 情况 2：用户问新闻，ToolNode 执行新闻工具 ========")
    print(run_case("002361 最近有什么新闻？"))

    print("======== 情况 3：模型不选择工具，直接回答 ========")
    print(run_case("002361 适合长期持有吗？"))


if __name__ == "__main__":
    demo_toolnode()
