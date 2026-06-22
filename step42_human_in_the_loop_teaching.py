"""第 42 步：看懂 Human-in-the-loop 人工审核。

Human-in-the-loop 是什么？

直译是：

人在流程里。

在多 Agent 投研系统里，它的意思是：

模型和 Agent 可以生成分析、风控报告、交易预案，
但某些关键决策不能自动继续，
必须经过人工确认。

比如：

1. 是否允许进入交易员节点。
2. 是否允许输出最终交易预案。
3. 是否允许把某次复盘写入长期记忆。
4. 是否允许真实下单。

尤其是交易相关系统，
Human-in-the-loop 很重要。

因为：

- 大模型可能幻觉。
- 数据源可能出错。
- 工具可能失败。
- Prompt 可能被误导。
- 交易动作有真实风险。

本文件做一个教学版：

风控报告
  ↓
交易员生成候选预案
  ↓
人工审核节点
  ├─ approve：通过，输出最终预案
  ├─ reject：拒绝，进入拒绝报告
  └─ revise：要求修改，输出修改后预案

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不等待真实用户输入。
- 用 simulated_human_decision 模拟人工审核。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


HumanDecision = Literal["approve", "reject", "revise"]


class HumanReviewState(TypedDict):
    """Human-in-the-loop 教学版状态。

    字段说明：

    - risk_report：风控报告。
    - draft_plan：交易员生成的候选预案。
    - human_decision：人工审核结果。
    - human_comment：人工审核意见。
    - final_plan：最终预案。
    - reject_report：拒绝原因报告。
    - step_log：执行日志。
    """

    risk_report: str
    draft_plan: str
    human_decision: HumanDecision
    human_comment: str
    final_plan: str
    reject_report: str
    step_log: list[str]


def create_initial_state(human_decision: HumanDecision) -> HumanReviewState:
    """创建初始状态。

    human_decision 用来模拟人工审核结果。

    真实系统里，这个值可能来自：

    - 页面按钮
    - 命令行输入
    - 企业微信/飞书审批
    - 人工标注系统
    """
    return {
        "risk_report": "风控报告：当前高风险，禁止追涨，只允许观察。",
        "draft_plan": "",
        "human_decision": human_decision,
        "human_comment": "",
        "final_plan": "",
        "reject_report": "",
        "step_log": [],
    }


def append_log(state: HumanReviewState, message: str) -> list[str]:
    """追加执行日志。"""
    return [*state["step_log"], message]


def trader_draft_node(state: HumanReviewState) -> dict:
    """交易员草案节点。

    交易员先生成一个“候选预案”。

    注意：
    这个预案还不是最终输出。
    它必须先经过 human_review_node。
    """
    draft_plan = (
        "候选交易预案：当前不主动开仓。"
        "若后续放量站回关键均线，并且风控下调风险等级，"
        "才重新讨论小仓位试探。"
    )
    return {
        "draft_plan": draft_plan,
        "step_log": append_log(state, "trader_draft_node：生成候选交易预案"),
    }


def human_review_node(state: HumanReviewState) -> dict:
    """人工审核节点。

    真实系统里，这里应该暂停流程，等待人审核。

    教学版里不暂停，
    而是读取 state["human_decision"] 模拟人工选择。

    三种结果：

    - approve：人工同意。
    - reject：人工拒绝。
    - revise：人工要求修改。
    """
    decision = state["human_decision"]

    if decision == "approve":
        comment = "人工审核：同意该防守型预案。"
    elif decision == "reject":
        comment = "人工审核：拒绝输出交易预案，原因是风险仍然过高。"
    else:
        comment = "人工审核：要求修改，必须更明确写出禁止追涨和仓位边界。"

    return {
        "human_comment": comment,
        "step_log": append_log(
            state,
            f"human_review_node：人工审核结果={decision}",
        ),
    }


def route_after_human_review(state: HumanReviewState) -> str:
    """人工审核后的路由。

    这是 Human-in-the-loop 的关键：

    后续流程不只由模型决定，
    还要由人工审核结果决定。
    """
    if state["human_decision"] == "approve":
        return "approve"

    if state["human_decision"] == "reject":
        return "reject"

    return "revise"


def approve_node(state: HumanReviewState) -> dict:
    """审核通过节点。"""
    return {
        "final_plan": (
            "最终预案已通过人工审核："
            f"{state['draft_plan']}"
        ),
        "step_log": append_log(state, "approve_node：输出最终预案"),
    }


def reject_node(state: HumanReviewState) -> dict:
    """审核拒绝节点。"""
    return {
        "reject_report": (
            "预案被人工拒绝，不输出交易方案。"
            f"审核意见：{state['human_comment']}"
        ),
        "step_log": append_log(state, "reject_node：停止输出交易预案"),
    }


def revise_node(state: HumanReviewState) -> dict:
    """审核要求修改节点。

    真实系统里，这里可能会把人工意见重新放进 Prompt，
    让交易员 Agent 重新生成预案。

    教学版里直接拼出修改后预案。
    """
    revised_plan = (
        "修改后最终预案：当前严格观望，禁止追涨，禁止新增仓位。"
        "仅允许记录观察条件：放量、站回关键均线、风险等级下调。"
        f"人工意见：{state['human_comment']}"
    )
    return {
        "final_plan": revised_plan,
        "step_log": append_log(state, "revise_node：根据人工意见修改预案"),
    }


def build_app():
    """构建 Human-in-the-loop 教学图。

    图结构：

    START
      ↓
    trader_draft
      ↓
    human_review
      ↓
    approve / reject / revise
      ↓
    END
    """
    graph = StateGraph(HumanReviewState)

    graph.add_node("trader_draft", trader_draft_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("approve", approve_node)
    graph.add_node("reject", reject_node)
    graph.add_node("revise", revise_node)

    graph.add_edge(START, "trader_draft")
    graph.add_edge("trader_draft", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        path_map={
            "approve": "approve",
            "reject": "reject",
            "revise": "revise",
        },
    )
    graph.add_edge("approve", END)
    graph.add_edge("reject", END)
    graph.add_edge("revise", END)

    return graph.compile()


def render_state(state: HumanReviewState) -> str:
    """渲染最终状态。"""
    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(state["step_log"], start=1)
    )
    return f"""风控报告：
{state["risk_report"]}

候选预案：
{state["draft_plan"]}

人工审核结果：
{state["human_decision"]}

人工审核意见：
{state["human_comment"]}

最终预案：
{state["final_plan"]}

拒绝报告：
{state["reject_report"]}

执行日志：
{step_log}
"""


def run_case(decision: HumanDecision) -> str:
    """运行一个人工审核案例。"""
    app = build_app()
    final_state = app.invoke(
        create_initial_state(decision),
        config={"recursion_limit": 10},
    )
    return f"""======== 人工审核案例：{decision} ========
{render_state(final_state)}
"""


def demo_human_in_the_loop() -> None:
    """演示 approve、reject、revise 三种人工审核结果。"""
    print(run_case("approve"))
    print(run_case("reject"))
    print(run_case("revise"))


if __name__ == "__main__":
    demo_human_in_the_loop()
