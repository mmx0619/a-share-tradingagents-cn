"""第 24 步：多 Agent 路由器。

前面我们已经有很多 Agent：

- market_agent：市场技术面分析
- news_agent：新闻事件分析
- summary_agent：综合汇总
- risk_agent：风控分析
- trader_agent：交易员预案
- realtime_summary_agent：实时综合分析
- realtime_risk_agent：实时风控
- realtime_trader_agent：实时交易员

你刚刚理解得很对：

上一个 Agent 的输出
  ↓
放进下一个 Agent 的 Prompt
  ↓
再调用大模型
  ↓
得到下一个 Agent 的输出

但是工程里还需要回答一个问题：

当前应该走哪个 Agent？

比如：

- 如果还没有 market_report，就应该先跑 market_agent。
- 如果已有 market_report 但没有 news_report，就应该跑 news_agent。
- 如果已有 market_report 和 news_report，就可以跑 summary_agent。
- 如果已有 summary_report，就可以跑 risk_agent。
- 如果已有 risk_report，就可以跑 trader_agent。

这个文件就是把这种“下一步去哪”写成代码。

后面接 LangGraph 时，
这些路由判断就会变成 graph.add_edge() 或 conditional_edges。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import step17_agent_state as agent_state


NodeName = Literal[
    "realtime_quote_node",
    "market_node",
    "news_node",
    "summary_node",
    "risk_node",
    "trader_node",
    "done",
]


@dataclass
class RouteDecision:
    """路由判断结果。

    字段说明：
    - next_node：下一步要执行哪个节点。
    - reason：为什么要去这个节点。

    这里先不用大模型判断路由。
    原因是：
    当前路由规则很明确，用代码判断更稳定。
    后面如果要做复杂路由，再考虑让大模型输出结构化路由结果。
    """

    next_node: NodeName
    reason: str


def decide_next_node(state: agent_state.TradingAgentState) -> RouteDecision:
    """根据当前 state 判断下一个节点。

    这个函数是整个路由器的核心。

    它不调用大模型。
    它只是检查 state 里哪些字段已经有值、哪些字段还没有。
    """
    if not state.realtime_quote_text:
        return RouteDecision(
            next_node="realtime_quote_node",
            reason="缺少实时行情快照，需要先获取实时行情。",
        )

    if not state.market_report:
        return RouteDecision(
            next_node="market_node",
            reason="缺少市场技术面报告，需要运行市场分析师 Agent。",
        )

    if not state.news_report:
        return RouteDecision(
            next_node="news_node",
            reason="缺少新闻面报告，需要运行新闻 Agent。",
        )

    if not state.summary_report:
        return RouteDecision(
            next_node="summary_node",
            reason="已有市场报告和新闻报告，可以运行综合汇总 Agent。",
        )

    if not state.risk_report:
        return RouteDecision(
            next_node="risk_node",
            reason="已有综合报告，可以运行风控 Agent。",
        )

    if not state.trader_plan:
        return RouteDecision(
            next_node="trader_node",
            reason="已有风控报告，可以运行交易员 Agent。",
        )

    return RouteDecision(
        next_node="done",
        reason="所有主要 Agent 输出都已完成，流程结束。",
    )


def render_route_decision(decision: RouteDecision) -> str:
    """把路由判断结果渲染成便于阅读的文本。"""
    return f"""下一节点：{decision.next_node}
原因：{decision.reason}
"""


def build_demo_state(stage: str) -> agent_state.TradingAgentState:
    """构造不同阶段的演示 state。

    参数 stage 用来模拟流程跑到了哪里：

    - empty：什么都没有。
    - realtime：已有实时行情。
    - market：已有市场报告。
    - news：已有新闻报告。
    - summary：已有综合报告。
    - risk：已有风控报告。
    - trader：已有交易员预案。
    """
    stage_order = [
        "empty",
        "realtime",
        "market",
        "news",
        "summary",
        "risk",
        "trader",
    ]
    if stage not in stage_order:
        raise ValueError(f"未知阶段：{stage}")

    current_index = stage_order.index(stage)

    def has_reached(target_stage: str) -> bool:
        """判断当前流程是否已经走到某个阶段。

        例如：
        - 当前是 risk。
        - target_stage 是 realtime。
        - 因为 risk 在 realtime 后面，所以返回 True。

        这比 `stage in {...}` 更适合学习，
        因为它直接表达了“当前进度是否已经到达某一步”。
        """
        target_index = stage_order.index(target_stage)
        return current_index >= target_index

    state = agent_state.TradingAgentState(
        symbol="002361",
        start_date="2026-01-01",
        end_date="2026-06-15",
    )

    if has_reached("realtime"):
        state.realtime_quote_text = "模拟实时行情快照。"

    if has_reached("market"):
        state.market_report = "模拟市场技术面报告。"

    if has_reached("news"):
        state.news_report = "模拟新闻面报告。"

    if has_reached("summary"):
        state.summary_report = "模拟综合汇总报告。"

    if has_reached("risk"):
        state.risk_report = "模拟风控报告。"

    if has_reached("trader"):
        state.trader_plan = "模拟交易员预案。"

    return state


def demo_agent_router() -> None:
    """演示不同 state 阶段会被路由到哪里。"""
    stages = ["empty", "realtime", "market", "news", "summary", "risk", "trader"]

    for stage in stages:
        state = build_demo_state(stage)
        decision = decide_next_node(state)
        print(f"当前阶段：{stage}")
        print(render_route_decision(decision))


if __name__ == "__main__":
    demo_agent_router()
