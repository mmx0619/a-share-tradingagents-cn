"""第 15 步：风控 Agent。

第 14 步已经完成综合投研汇总：

技术面 Agent
  ↓
新闻面 Agent
  ↓
综合投研汇总 Agent

当前文件做第 15 件事：

综合投研汇总报告
  ↓
风控 Prompt
  ↓
大模型
  ↓
风控报告

风控 Agent 的定位：
它不是为了找机会，
而是为了限制风险、识别禁止动作、给交易员 Agent 设置约束。

在 TradingAgents 这类多智能体系统里，
风控 Agent 的价值是“刹车”和“边界”。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step14_research_summary_agent as summary_agent


@dataclass
class RiskControlResult:
    """风控 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - research_summary：第 14 步生成的综合投研汇总报告。
    - prompt：发送给风控 Agent 的完整 Prompt。
    - provider：模型平台，比如 deepseek。
    - model：具体模型名称。
    - risk_report：风控 Agent 输出的报告。
    """

    symbol: str
    research_summary: str
    prompt: str
    provider: str
    model: str | None
    risk_report: str


def build_risk_control_prompt(symbol: str, research_summary: str) -> str:
    """生成风控 Agent 的 Prompt。

    风控 Agent 只基于综合投研报告工作。
    它不重新分析原始行情和新闻，
    只负责把已有结论转成风险约束。
    """
    return f"""你是一名 A 股多智能体投研系统里的风控 Agent。

你现在收到的是综合投研汇总报告。

你的任务不是寻找交易机会，
而是识别风险、约束交易员 Agent、给出风险边界。

请遵守以下要求：

1. 只基于我提供的综合投研汇总报告分析，不要编造新数据。
2. 不要输出确定性的买入或卖出指令。
3. 如果报告中出现高换手、龙虎榜、短线剧烈波动、跌破均线、流动性风险，要提高风险等级。
4. 必须明确列出“禁止动作”，比如禁止追涨、禁止重仓、禁止无止损参与。
5. 必须给出后续观察条件，比如缩量止跌、放量反包、跌破关键位等。
6. 仓位建议只能给范围和约束，不要给确定交易指令。

输出格式：

## 风险结论
用 2-4 句话总结风险状态。

## 风险等级
从 低 / 中 / 高 / 极高 中选择一个，并说明原因。

## 主要风险来源
列出风险来自技术面、新闻面、资金面还是流动性。

## 禁止动作
列出当前不应该做的事情。

## 观察条件
列出后续如果要重新评估，需要观察哪些条件。

## 仓位约束
只给风险约束，不给确定交易指令。

## 给交易员 Agent 的边界
告诉交易员 Agent 在什么条件下才允许继续讨论交易方案。

股票代码：
{symbol}

综合投研汇总报告：
{research_summary}
"""


def run_risk_control_agent(
    symbol: str,
    research_summary: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> RiskControlResult:
    """运行风控 Agent。"""
    prompt = build_risk_control_prompt(
        symbol=symbol,
        research_summary=research_summary,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return RiskControlResult(
        symbol=symbol,
        research_summary=research_summary,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        risk_report=response.text,
    )


def run_full_risk_control_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> RiskControlResult:
    """运行完整风控链路。

    当前完整链路：
    1. 第 14 步先生成综合投研汇总报告。
    2. 第 15 步把综合报告交给风控 Agent。
    3. 风控 Agent 输出风险等级、禁止动作和交易边界。
    """
    summary_result = summary_agent.run_full_research_summary_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )

    return run_risk_control_agent(
        symbol=summary_result.symbol,
        research_summary=summary_result.summary_text,
        provider=provider,
        model=model,
    )


def render_risk_control_result(result: RiskControlResult) -> str:
    """把风控 Agent 的结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 综合投研汇总报告 ========
{result.research_summary}

======== 风控 Agent 报告 ========
{result.risk_report}
"""


def demo_risk_control_pipeline() -> None:
    """演示完整风控链路。

    默认继续使用 002361。
    这类高换手、龙虎榜、短线大波动股票，
    很适合测试风控 Agent 是否能起到“刹车”作用。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_full_risk_control_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_risk_control_result(result))


if __name__ == "__main__":
    demo_risk_control_pipeline()
