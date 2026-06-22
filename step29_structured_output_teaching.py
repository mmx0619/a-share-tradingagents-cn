"""第 29 步：单文件看懂结构化输出、校验和兜底。

第 28 步讲的是：

上一个 Agent 的输出
  ↓
放进下一个 Agent 的 Prompt
  ↓
再交给大模型
  ↓
得到下一个 Agent 的输出

但这里有一个工程问题：

如果大模型只返回一段普通文本，
程序很难稳定判断下一步该做什么。

例如模型返回：

    我觉得风险挺高，先别追。

人能看懂。
但程序很难稳定判断：

- 风险等级到底是 high 还是 medium？
- 是否允许交易？
- 下一节点应该去 risk_node 还是 trader_node？

所以真实工程里经常要求模型返回结构化结果。

结构化结果可以理解成：

模型不能随便写一段散文，
而是要按我们规定好的字段返回。

例如：

{
    "risk_level": "high",
    "allow_trade": false,
    "next_node": "trader_node",
    "reason": "短线波动较大，禁止追涨"
}

本文件专门演示：

1. 什么是 schema。
2. 什么是枚举值。
3. 如何解析模型返回的 JSON。
4. 如何校验字段和值。
5. 如果模型返回错了，如何兜底成合法对象。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 只用 Python 标准库和 dataclass。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal


# Literal 表示“只能从这些固定值里选”。
#
# 例如 RiskLevel 只能是：
# - "low"
# - "medium"
# - "high"
#
# 如果模型返回：
# - "比较高"
# - "偏高"
# - "严重"
#
# 这些虽然人能理解，
# 但程序不会接受。
RiskLevel = Literal["low", "medium", "high"]
NextNode = Literal["summary_node", "risk_node", "trader_node", "done", "fallback"]


@dataclass
class RiskDecision:
    """风控结构化输出对象。

    这个 dataclass 就是教学版 schema。

    你可以把它理解成：

    程序要求模型最终必须给我这些字段：

    - risk_level：风险等级，只能是 low / medium / high。
    - allow_trade：是否允许进入交易讨论，只能是 True / False。
    - next_node：下一步节点，只能是固定节点名。
    - reason：原因说明，必须是字符串。

    注意：
    dataclass 本身不会自动强校验 Literal。
    所以下面还会写 validate_risk_decision_dict() 手动校验。
    """

    risk_level: RiskLevel
    allow_trade: bool
    next_node: NextNode
    reason: str


def build_risk_prompt(summary_report: str) -> str:
    """构造要求模型返回固定 JSON 的 Prompt。

    真实调用大模型时，
    你会把这个 Prompt 发给模型。

    这里的重点是：
    Prompt 里明确告诉模型：

    1. 必须返回 JSON。
    2. 必须包含哪些字段。
    3. 每个字段允许哪些值。
    4. 不要输出 JSON 以外的解释文字。

    但要记住：
    只靠 Prompt 不能 100% 保证模型听话。
    所以后面还必须有程序校验。
    """
    return f"""你是 A 股多智能体系统里的风控 Agent。

请根据综合研究报告，返回一个严格 JSON。

只能返回 JSON，不要输出 Markdown，不要输出解释文字。

JSON 字段必须是：

{{
  "risk_level": "low | medium | high",
  "allow_trade": true 或 false,
  "next_node": "summary_node | risk_node | trader_node | done | fallback",
  "reason": "你的原因说明"
}}

字段要求：

1. risk_level 只能是 low、medium、high。
2. allow_trade 必须是布尔值 true 或 false。
3. next_node 只能是 summary_node、risk_node、trader_node、done、fallback。
4. reason 必须是字符串。

综合研究报告：
{summary_report}
"""


def mock_llm_good(prompt: str) -> str:
    """模拟模型正常返回合法 JSON。"""
    return json.dumps(
        {
            "risk_level": "high",
            "allow_trade": False,
            "next_node": "trader_node",
            "reason": "短线波动较大，禁止主动追涨，只允许交易员生成防守型预案。",
        },
        ensure_ascii=False,
    )


def mock_llm_bad_text(prompt: str) -> str:
    """模拟模型没有按 JSON 返回，只返回普通文本。"""
    return "我觉得风险挺高，先别追，后面让交易员保守一点。"


def mock_llm_bad_enum(prompt: str) -> str:
    """模拟模型返回了 JSON，但字段值不是我们规定的枚举值。"""
    return json.dumps(
        {
            "risk_level": "偏高",
            "allow_trade": "不允许",
            "next_node": "交易员节点",
            "reason": "模型用了中文近义词，但程序不接受。",
        },
        ensure_ascii=False,
    )


def parse_json_object(model_text: str) -> dict:
    """把模型返回文本解析成 dict。

    如果模型返回的不是合法 JSON，
    json.loads() 会抛出 JSONDecodeError。

    这里不在本函数里吞掉异常，
    而是交给上层 parse_risk_decision() 统一兜底。
    """
    parsed = json.loads(model_text)
    if not isinstance(parsed, dict):
        raise ValueError("模型返回的是 JSON，但不是对象。")
    return parsed


def validate_risk_decision_dict(data: dict) -> RiskDecision:
    """校验 dict 是否符合 RiskDecision schema。

    这一步是关键。

    即使模型返回了合法 JSON，
    也不代表它符合程序要求。

    例如：

    {
        "risk_level": "偏高"
    }

    这是合法 JSON，
    但不是合法业务对象。

    所以必须继续校验字段和值。
    """
    allowed_risk_levels = {"low", "medium", "high"}
    allowed_next_nodes = {"summary_node", "risk_node", "trader_node", "done", "fallback"}

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


def fallback_risk_decision(error_message: str) -> RiskDecision:
    """兜底成合法对象。

    你前面问过：

    如果模型没有正常返回，
    下一段程序还要靠这个格式触发，
    那怎么办？

    答案是：

    不能兜底成普通文本。
    必须兜底成合法对象。

    这里我们返回一个保守的合法 RiskDecision：

    - risk_level = high
    - allow_trade = False
    - next_node = fallback
    - reason = 说明为什么进入兜底

    这样后续程序仍然能继续运行。
    """
    return RiskDecision(
        risk_level="high",
        allow_trade=False,
        next_node="fallback",
        reason=f"模型输出解析或校验失败，进入保守兜底。错误：{error_message}",
    )


def parse_risk_decision(model_text: str) -> RiskDecision:
    """解析模型输出，失败时返回合法兜底对象。

    这就是完整流程：

    模型文本
      ↓
    JSON 解析
      ↓
    字段和值校验
      ↓
    成功：返回 RiskDecision
      ↓
    失败：返回 fallback RiskDecision

    注意：
    无论成功还是失败，
    本函数最终都会返回 RiskDecision。

    这就是“必须兜底成合法 JSON/对象”的代码含义。
    """
    try:
        data = parse_json_object(model_text)
        return validate_risk_decision_dict(data)
    except Exception as error:
        return fallback_risk_decision(str(error))


def run_case(case_name: str, model_text: str) -> str:
    """运行一个测试案例，并渲染结果。"""
    decision = parse_risk_decision(model_text)
    return f"""案例：{case_name}

模型原始输出：
{model_text}

程序解析后的合法对象：
risk_level = {decision.risk_level}
allow_trade = {decision.allow_trade}
next_node = {decision.next_node}
reason = {decision.reason}
"""


def demo_structured_output() -> None:
    """演示三种模型输出情况。"""
    summary_report = (
        "综合研究报告：技术面偏弱，新闻面显示短线资金博弈强，"
        "实时小幅反弹暂时不能证明趋势反转。"
    )
    prompt = build_risk_prompt(summary_report)

    print("======== 发给模型的 Prompt ========")
    print(prompt)

    print("======== 情况 1：模型返回合法 JSON ========")
    print(run_case("合法 JSON", mock_llm_good(prompt)))

    print("======== 情况 2：模型返回普通文本 ========")
    print(run_case("普通文本", mock_llm_bad_text(prompt)))

    print("======== 情况 3：模型返回 JSON，但枚举值不合法 ========")
    print(run_case("非法枚举", mock_llm_bad_enum(prompt)))


if __name__ == "__main__":
    demo_structured_output()
