"""第 14 步：综合投研汇总 Agent。

前面已经有两个独立 Agent：

- step10_real_market_agent_pipeline.py
  负责真实行情、技术指标、市场快照、技术面分析。

- step13_news_agent.py
  负责新闻获取、事件抽取、新闻面分析。

当前文件做第 14 件事：

技术面报告
  ↓
新闻面报告
  ↓
综合投研 Prompt
  ↓
大模型
  ↓
综合投研摘要

注意：
这个文件仍然不直接给“买入/卖出”指令。
它的定位更像 TradingAgents 里的研究经理：
把不同分析师的观点汇总，找出共振、冲突、风险和下一步需要补充的信息。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step10_real_market_agent_pipeline as market_pipeline
import step11_stock_news as stock_news
import step12_news_event_extractor as event_extractor
import step13_news_agent as news_agent


@dataclass
class ResearchSummaryResult:
    """综合投研汇总结果。

    字段说明：
    - symbol：股票代码。
    - market_report：技术面 Agent 报告。
    - news_report：新闻面 Agent 报告。
    - prompt：发送给综合 Agent 的完整提示词。
    - provider：模型平台，比如 deepseek。
    - model：具体模型名称。
    - summary_text：综合投研摘要。
    """

    symbol: str
    market_report: str
    news_report: str
    prompt: str
    provider: str
    model: str | None
    summary_text: str


def build_research_summary_prompt(
    symbol: str,
    market_report: str,
    news_report: str,
) -> str:
    """生成综合投研汇总 Prompt。

    这个 Prompt 的重点：
    不是让模型重新分析所有原始数据，
    而是让它比较两个 Agent 的结论。

    也就是：
    技术面和新闻面有没有互相支持？
    如果互相冲突，应该如何提醒后续 Agent？
    """
    return f"""你是一名 A 股多智能体投研系统的研究经理。

现在你已经收到两个分析师 Agent 的报告：

1. 技术面市场分析师报告
2. 新闻事件分析师报告

你的任务不是直接给买卖建议，
而是对两个 Agent 的观点做综合归纳，判断它们之间是共振、冲突还是互补。

请遵守以下要求：

1. 只基于我提供的两份报告总结，不要编造新的行情或新闻。
2. 不要输出“买入、卖出、持有”等最终交易指令。
3. 明确指出哪些结论是技术面支持的，哪些结论是新闻面支持的。
4. 如果技术面偏强但新闻面提示高风险，要明确说明冲突。
5. 如果技术面和新闻面都提示风险，要提高风险等级。
6. 输出要适合后续交易员 Agent 和风控 Agent 阅读。

输出格式：

## 综合结论
用 2-4 句话总结当前综合状态。

## 多 Agent 共识
说明技术面和新闻面在哪些地方互相支持。

## 多 Agent 分歧
说明技术面和新闻面在哪些地方互相冲突或信息不足。

## 风险等级
给出低 / 中 / 高 / 极高 四档之一，并说明原因。

## 给后续 Agent 的任务
告诉交易员 Agent、风控 Agent 下一步应该重点检查什么。

股票代码：
{symbol}

技术面市场分析师报告：
{market_report}

新闻事件分析师报告：
{news_report}
"""


def run_research_summary_agent(
    symbol: str,
    market_report: str,
    news_report: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> ResearchSummaryResult:
    """运行综合投研汇总 Agent。"""
    prompt = build_research_summary_prompt(
        symbol=symbol,
        market_report=market_report,
        news_report=news_report,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return ResearchSummaryResult(
        symbol=symbol,
        market_report=market_report,
        news_report=news_report,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        summary_text=response.text,
    )


def run_full_research_summary_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> ResearchSummaryResult:
    """运行完整综合投研链路。

    当前完整链路：
    1. 第 10 步生成技术面报告。
    2. 第 11 步获取新闻。
    3. 第 12 步抽取新闻事件。
    4. 第 13 步生成新闻面报告。
    5. 第 14 步汇总技术面和新闻面。
    """
    market_result = market_pipeline.run_real_market_agent_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        model=model,
    )

    news_items = stock_news.get_stock_news(symbol, max_items=news_max_items)
    news_events = event_extractor.extract_news_events(news_items)
    news_result = news_agent.run_news_agent(
        symbol=symbol,
        news_events=news_events,
        provider=provider,
        model=model,
    )

    return run_research_summary_agent(
        symbol=market_result.symbol,
        market_report=market_result.report_text,
        news_report=news_result.report_text,
        provider=provider,
        model=model,
    )


def render_research_summary_result(result: ResearchSummaryResult) -> str:
    """把综合投研汇总结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 技术面市场分析师报告 ========
{result.market_report}

======== 新闻事件分析师报告 ========
{result.news_report}

======== 综合投研汇总报告 ========
{result.summary_text}
"""


def demo_research_summary_pipeline() -> None:
    """演示完整综合投研汇总链路。

    默认用 002361，是因为它同时有：
    - 真实行情波动
    - 龙虎榜新闻
    - 高换手事件
    - 题材异动

    这种股票更适合观察多 Agent 之间的共识和冲突。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_full_research_summary_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_research_summary_result(result))


if __name__ == "__main__":
    demo_research_summary_pipeline()
