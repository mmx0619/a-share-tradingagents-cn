"""第 40 步：看懂 LangGraph Checkpoint 断点保存。

Checkpoint 是什么？

你可以理解成：

程序每跑完一步，
LangGraph 把当前 state 保存下来。

这样如果流程很长，或者中途失败，
后面可以根据同一个 thread_id 找回之前的状态。

在多 Agent 项目里，Checkpoint 很重要。

原因：

1. 多 Agent 流程可能很长。
2. 中间可能要调用真实大模型，成本比较高。
3. 中间可能要调用行情、新闻、公告等外部工具，容易失败。
4. 如果失败后从头再跑，会浪费时间和 API 成本。
5. 有了 checkpoint，就可以保存每一步状态，方便恢复和调试。

本文件只做教学版：

- 不联网。
- 不调用真实大模型。
- 使用 LangGraph 自带 MemorySaver。
- 演示同一个 thread_id 下，状态如何保存。

注意：
MemorySaver 是内存版 checkpoint。
程序退出后，保存内容也会消失。

真实项目里可以换成 SQLite、Postgres、Redis 等持久化存储。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class CheckpointState(TypedDict):
    """图状态。

    step：当前执行到第几步。
    report：模拟 Agent 逐步写出的报告。
    """

    step: int
    report: str


def market_node(state: CheckpointState) -> dict:
    """市场分析节点。

    模拟第一个 Agent 写入市场分析结果。
    """
    return {
        "step": state["step"] + 1,
        "report": state["report"] + "\n市场分析：短线价格偏弱。",
    }


def news_node(state: CheckpointState) -> dict:
    """新闻分析节点。

    模拟第二个 Agent 继续追加新闻分析结果。
    """
    return {
        "step": state["step"] + 1,
        "report": state["report"] + "\n新闻分析：龙虎榜和高换手较多。",
    }


def risk_node(state: CheckpointState) -> dict:
    """风控节点。

    模拟第三个 Agent 追加风控结论。
    """
    return {
        "step": state["step"] + 1,
        "report": state["report"] + "\n风控结论：维持高风险，禁止追涨。",
    }


def build_app():
    """构建带 checkpoint 的 LangGraph。

    重点在这里：

    checkpointer = MemorySaver()
    graph.compile(checkpointer=checkpointer)

    这样图运行时就会保存状态。
    """
    graph = StateGraph(CheckpointState)

    graph.add_node("market", market_node)
    graph.add_node("news", news_node)
    graph.add_node("risk", risk_node)

    graph.add_edge(START, "market")
    graph.add_edge("market", "news")
    graph.add_edge("news", "risk")
    graph.add_edge("risk", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app


def run_checkpoint_demo() -> str:
    """运行 checkpoint 演示。

    thread_id 是 checkpoint 的关键。

    你可以把 thread_id 理解成：

    这一次对话或这一次任务的 ID。

    同一个 thread_id 下，
    LangGraph 知道这些状态属于同一次流程。
    """
    app = build_app()

    config = {
        "configurable": {
            "thread_id": "demo-thread-002361",
        }
    }

    initial_state: CheckpointState = {
        "step": 0,
        "report": "股票代码：002361",
    }

    final_state = app.invoke(initial_state, config=config)

    # get_state(config) 可以读取当前 thread_id 下保存的最新状态。
    saved_state = app.get_state(config)

    return f"""======== 最终返回的 final_state ========
step = {final_state["step"]}
report =
{final_state["report"]}

======== checkpoint 保存的最新状态 ========
values =
{saved_state.values}

next =
{saved_state.next}

config =
{saved_state.config}
"""


if __name__ == "__main__":
    print(run_checkpoint_demo())
