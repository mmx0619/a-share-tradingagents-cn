"""第 21 步：实时行情感知的综合投研 Agent。

第 14 步的综合 Agent 只看：

- 技术面市场分析师报告
- 新闻事件分析师报告

第 20 步已经把实时行情快照放进了 TradingAgentState，
但综合 Agent 还没有真正读取它。

当前文件做第 21 件事：

实时行情快照
  +
技术面报告
  +
新闻面报告
  ↓
实时感知综合 Prompt
  ↓
大模型
  ↓
实时感知综合投研报告

这一步要解决的问题：

历史日线可能显示弱势，
但当前实时行情可能出现反弹。

这时候综合 Agent 不能简单地说“弱势”，
而应该判断：

- 实时反弹是否改变了日线弱势？
- 是真实修复，还是下跌后的技术性反抽？
- 是否需要风控 Agent 更新风险边界？
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step17_agent_state as agent_state
import step20_realtime_stateful_workflow as realtime_workflow


@dataclass
class RealtimeSummaryResult:
    """实时行情感知综合 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - realtime_quote_text：实时行情快照。
    - market_report：技术面报告。
    - news_report：新闻面报告。
    - prompt：发送给模型的完整 Prompt。
    - provider：模型平台。
    - model：具体模型名称。
    - summary_text：实时感知综合报告。
    """

    symbol: str
    realtime_quote_text: str
    market_report: str
    news_report: str
    prompt: str
    provider: str
    model: str | None
    summary_text: str


def build_realtime_summary_prompt(
    symbol: str,
    realtime_quote_text: str,
    market_report: str,
    news_report: str,
) -> str:
    """生成实时行情感知综合 Prompt。

    这个 Prompt 会把实时行情快照放在最前面。
    目的是让模型先知道当前盘面状态，
    再去对照日线技术面和新闻面。
    """
    return f"""你是一名 A 股多智能体投研系统的实时研究经理。

你现在收到三类信息：

1. 实时/近实时行情快照
2. 技术面市场分析师报告
3. 新闻事件分析师报告

你的任务不是直接给买卖建议，
而是判断“实时行情”是否改变了前面技术面和新闻面的判断。

请遵守以下要求：

1. 只基于我提供的信息分析，不要编造新的行情或新闻。
2. 不要输出“买入、卖出、持有”等最终交易指令。
3. 必须区分历史日线结论和实时盘面变化。
4. 如果实时行情只是小幅反弹，但技术面和新闻面仍然高风险，要明确说明不能轻易认为趋势反转。
5. 如果实时行情和历史弱势结论冲突，要说明冲突点以及后续需要观察的数据。
6. 输出要适合后续风控 Agent 和交易员 Agent 阅读。

输出格式：

## 实时综合结论
用 2-4 句话说明实时行情是否改变原有判断。

## 实时行情信号
说明实时快照中的最新价、涨跌幅、成交额等有什么意义。

## 与技术面报告的关系
说明实时行情和日线技术面是共振、修复还是冲突。

## 与新闻面报告的关系
说明实时行情和新闻事件风险是否一致。

## 风险等级调整建议
给出是否建议风控 Agent 上调、维持或下调风险等级，并说明原因。

## 给后续 Agent 的任务
告诉风控 Agent、交易员 Agent 下一步应该重点检查什么。

股票代码：
{symbol}

实时行情快照：
{realtime_quote_text}

技术面市场分析师报告：
{market_report}

新闻事件分析师报告：
{news_report}
"""


def run_realtime_summary_agent(
    symbol: str,
    realtime_quote_text: str,
    market_report: str,
    news_report: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> RealtimeSummaryResult:
    """运行实时行情感知综合 Agent。"""
    prompt = build_realtime_summary_prompt(
        symbol=symbol,
        realtime_quote_text=realtime_quote_text,
        market_report=market_report,
        news_report=news_report,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return RealtimeSummaryResult(
        symbol=symbol,
        realtime_quote_text=realtime_quote_text,
        market_report=market_report,
        news_report=news_report,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        summary_text=response.text,
    )


def run_realtime_summary_from_state(
    state: agent_state.TradingAgentState,
    provider: str = "deepseek",
    model: str | None = None,
) -> RealtimeSummaryResult:
    """从 TradingAgentState 中读取信息并运行实时综合 Agent。"""
    if not state.realtime_quote_text:
        raise ValueError("缺少 realtime_quote_text，不能运行实时综合 Agent。")
    if not state.market_report:
        raise ValueError("缺少 market_report，不能运行实时综合 Agent。")
    if not state.news_report:
        raise ValueError("缺少 news_report，不能运行实时综合 Agent。")

    return run_realtime_summary_agent(
        symbol=state.symbol,
        realtime_quote_text=state.realtime_quote_text,
        market_report=state.market_report,
        news_report=state.news_report,
        provider=provider,
        model=model,
    )


def render_realtime_summary_result(result: RealtimeSummaryResult) -> str:
    """把实时综合结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 实时行情快照 ========
{result.realtime_quote_text}

======== 技术面市场分析师报告 ========
{result.market_report}

======== 新闻事件分析师报告 ========
{result.news_report}

======== 实时感知综合投研报告 ========
{result.summary_text}
"""


def demo_realtime_summary_agent() -> None:
    """演示实时行情感知综合 Agent。

    为了避免重复跑风控和交易员，
    这里先用第 20 步跑到完整 state，
    然后只额外运行一次实时综合 Agent。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    state = realtime_workflow.run_realtime_stateful_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    result = run_realtime_summary_from_state(
        state=state,
        provider=provider,
        model=model,
    )
    print(render_realtime_summary_result(result))


if __name__ == "__main__":
    demo_realtime_summary_agent()
