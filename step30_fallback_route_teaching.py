"""第 30 步：看懂 fallback 返回后，程序如何继续运行。

第 29 步里有一个保底输出：

RiskDecision(
    risk_level="high",
    allow_trade=False,
    next_node="fallback",
    reason="模型输出解析或校验失败，进入保守兜底。"
)

你刚刚的理解是对的：

当模型没有返回我们需要的格式，
程序就会返回这个保底对象。

但还有一个问题：

返回 next_node="fallback" 以后，
程序又怎么真的进入 fallback 节点？

本文件专门演示这件事。

核心流程：

模型输出
  ↓
解析和校验
  ↓
得到 RiskDecision
  ↓
把 RiskDecision 写入 state
  ↓
路由函数读取 state["risk_decision"].next_node
  ↓
如果 next_node 是 fallback，就进入 fallback_node

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 用 LangGraph 演示 fallback 路由。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


RiskLevel = Literal["low", "medium", "high"]
NextNode = Literal["trader_node", "done", "fallback"]


@dataclass
class RiskDecision:
    """风控结构化结果。

    这个对象是后续路由的依据。

    尤其是 next_node 字段：

    - trader_node：进入交易员节点。
    - done：流程直接结束。
    - fallback：进入兜底节点。
    """

    risk_level: RiskLevel
    allow_trade: bool
    next_node: NextNode
    reason: str


class FallbackRouteState(TypedDict):
    """LangGraph 共享状态。

    字段说明：

    - model_text：模拟模型原始输出。
    - risk_decision：解析后的结构化对象。
    - trader_plan：交易员预案。
    - fallback_report：兜底节点生成的报告。
    - step_log：执行日志。
    """

    model_text: str
    risk_decision: RiskDecision | None
    trader_plan: str
    fallback_report: str
    step_log: list[str]


def create_initial_state(model_text: str) -> FallbackRouteState:
    """创建初始状态。"""
    return {
        "model_text": model_text,
        "risk_decision": None,
        "trader_plan": "",
        "fallback_report": "",
        "step_log": [],
    }


def append_log(state: FallbackRouteState, message: str) -> list[str]:
    """追加执行日志。"""
    return [*state["step_log"], message]


def fallback_risk_decision(error_message: str) -> RiskDecision:
    """生成保守兜底对象。

    这里和第 29 步一样：

    只要模型输出不能被程序稳定识别，
    就不要冒险进入正常交易节点。

    所以 next_node 固定为 fallback。
    """
    return RiskDecision(
        risk_level="high",
        allow_trade=False,
        next_node="fallback",
        reason=f"模型输出解析或校验失败，进入保守兜底。错误：{error_message}",
    )


def validate_risk_decision(data: dict) -> RiskDecision:
    """校验模型返回的 dict 是否符合要求。"""
    allowed_risk_levels = {"low", "medium", "high"}
    allowed_next_nodes = {"trader_node", "done", "fallback"}

    risk_level = data.get("risk_level")
    allow_trade = data.get("allow_trade")
    next_node = data.get("next_node")
    reason = data.get("reason")

    if risk_level not in allowed_risk_levels:
        raise ValueError(f"risk_level 不合法：{risk_level}")

    if not isinstance(allow_trade, bool):
        raise ValueError(f"allow_trade 必须是布尔值：{allow_trade}")

    if next_node not in allowed_next_nodes:
        raise ValueError(f"next_node 不合法：{next_node}")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason 必须是非空字符串。")

    return RiskDecision(
        risk_level=risk_level,
        allow_trade=allow_trade,
        next_node=next_node,
        reason=reason,
    )


def parse_risk_decision(model_text: str) -> RiskDecision:
    """解析模型输出，失败则返回保底对象。"""
    try:
        data = json.loads(model_text)
        if not isinstance(data, dict):
            raise ValueError("模型返回的是 JSON，但不是对象。")
        return validate_risk_decision(data)
    except Exception as error:
        return fallback_risk_decision(str(error))


def risk_parse_node(state: FallbackRouteState) -> dict:
    """风控解析节点。

    这个节点负责：

    1. 读取模型原始输出 model_text。
    2. 调用 parse_risk_decision()。
    3. 得到 RiskDecision。
    4. 把 RiskDecision 写入 state。

    注意：
    不管模型输出正常还是异常，
    parse_risk_decision() 都会返回 RiskDecision。

    所以后续路由一定有对象可读。
    """
    decision = parse_risk_decision(state["model_text"])
    return {
        "risk_decision": decision,
        "step_log": append_log(
            state,
            f"risk_parse_node：解析完成，next_node={decision.next_node}",
        ),
    }


def trader_node(state: FallbackRouteState) -> dict:
    """正常交易员节点。

    只有当 risk_decision.next_node == "trader_node" 时，
    LangGraph 才会进入这个节点。
    """
    decision = state["risk_decision"]
    if decision is None:
        raise ValueError("缺少 risk_decision，不能运行 trader_node。")

    plan = (
        "正常交易员预案：风控结构化输出合法，"
        f"风险等级={decision.risk_level}，"
        f"是否允许交易={decision.allow_trade}。"
    )
    return {
        "trader_plan": plan,
        "step_log": append_log(state, "trader_node：进入正常交易员节点"),
    }


def fallback_node(state: FallbackRouteState) -> dict:
    """兜底节点。

    只有当 risk_decision.next_node == "fallback" 时，
    LangGraph 才会进入这个节点。

    兜底节点一般做保守处理：

    - 不继续生成激进交易计划。
    - 标记本次模型输出异常。
    - 要求人工复核或重新请求模型。
    - 给出安全的默认结论。
    """
    decision = state["risk_decision"]
    if decision is None:
        raise ValueError("缺少 risk_decision，不能运行 fallback_node。")

    report = (
        "兜底报告：模型输出没有通过解析或校验。"
        "系统进入保守模式，不允许交易，建议人工复核。"
        f"原因：{decision.reason}"
    )
    return {
        "fallback_report": report,
        "step_log": append_log(state, "fallback_node：进入兜底节点"),
    }


def route_after_risk_parse(state: FallbackRouteState) -> str:
    """解析节点之后的路由函数。

    这个函数就是回答你的问题：

    return RiskDecision(next_node="fallback") 后，
    程序怎么继续运行？

    答案：

    1. risk_parse_node 把 RiskDecision 写入 state。
    2. LangGraph 调用这个路由函数。
    3. 这个函数读取 state["risk_decision"].next_node。
    4. 如果值是 "fallback"，就返回 "fallback_node"。
    5. LangGraph 根据 path_map 进入 fallback_node。
    """
    decision = state["risk_decision"]
    if decision is None:
        return "fallback_node"

    if decision.next_node == "trader_node":
        return "trader_node"

    if decision.next_node == "fallback":
        return "fallback_node"

    return "done"


def build_graph():
    """构建 LangGraph。

    图结构：

    START
      ↓
    risk_parse_node
      ↓
    route_after_risk_parse
      ↓
    trader_node / fallback_node / END
      ↓
    END
    """
    graph = StateGraph(FallbackRouteState)

    graph.add_node("risk_parse_node", risk_parse_node)
    graph.add_node("trader_node", trader_node)
    graph.add_node("fallback_node", fallback_node)

    graph.add_edge(START, "risk_parse_node")

    graph.add_conditional_edges(
        "risk_parse_node",
        route_after_risk_parse,
        path_map={
            "trader_node": "trader_node",
            "fallback_node": "fallback_node",
            "done": END,
        },
    )

    graph.add_edge("trader_node", END)
    graph.add_edge("fallback_node", END)

    return graph.compile()


def run_case(case_name: str, model_text: str) -> str:
    """运行一个案例。"""
    app = build_graph()
    final_state = app.invoke(
        create_initial_state(model_text),
        config={"recursion_limit": 10},
    )
    decision = final_state["risk_decision"]
    if decision is None:
        raise ValueError("流程结束后仍然没有 risk_decision。")

    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(final_state["step_log"], start=1)
    )
    return f"""案例：{case_name}

模型原始输出：
{model_text}

解析后的 next_node：
{decision.next_node}

解析后的 reason：
{decision.reason}

执行日志：
{step_log}

交易员预案：
{final_state["trader_plan"]}

兜底报告：
{final_state["fallback_report"]}
"""


def demo_fallback_route() -> None:
    """演示正常路由和 fallback 路由。"""
    good_model_text = json.dumps(
        {
            "risk_level": "medium",
            "allow_trade": True,
            "next_node": "trader_node",
            "reason": "模型返回格式合法，可以进入交易员节点生成条件式预案。",
        },
        ensure_ascii=False,
    )

    bad_model_text = "我觉得风险挺高，先别追，后面让交易员保守一点。"

    print("======== 情况 1：模型返回合法结构，进入 trader_node ========")
    print(run_case("正常进入交易员节点", good_model_text))

    print("======== 情况 2：模型返回普通文本，进入 fallback_node ========")
    print(run_case("进入兜底节点", bad_model_text))


if __name__ == "__main__":
    demo_fallback_route()
