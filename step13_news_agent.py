"""第 13 步：新闻事件分析 Agent。

第 11 步负责获取新闻原文。
第 12 步负责把新闻原文抽取成事件信号。

当前文件做第 13 件事：

新闻事件信号
  ↓
新闻分析 Prompt
  ↓
大模型
  ↓
新闻面分析报告

这个 Agent 的重点不是复述新闻，
而是判断这些新闻事件对股票有什么分析意义：

- 是普通资讯，还是值得关注的交易事件？
- 是基本面事件，还是短线资金博弈？
- 是题材驱动，还是风险释放？
- 后续交易员 Agent 应该重点关注什么？
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step11_stock_news as stock_news
import step12_news_event_extractor as event_extractor


@dataclass
class NewsAgentResult:
    """新闻 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - provider：模型平台，比如 deepseek。
    - model：具体模型名称。
    - events_text：发送给模型前整理好的新闻事件文本。
    - prompt：发送给模型的完整 Prompt。
    - report_text：模型生成的新闻面分析报告。
    """

    symbol: str
    provider: str
    model: str | None
    events_text: str
    prompt: str
    report_text: str


def build_news_agent_prompt(symbol: str, events_text: str) -> str:
    """生成新闻 Agent 的 Prompt。

    这里输入的不是新闻全文，而是第 12 步抽取后的事件信号。
    这样可以减少噪音，让模型重点分析真正有用的信息。
    """
    return f"""你是一名 A 股新闻事件分析师。

你的任务是根据给定的新闻事件信号，判断这些事件对股票短期交易和风险有什么意义。

请遵守以下要求：

1. 只基于我提供的新闻事件信号分析，不要编造不存在的新闻。
2. 不要直接给出最终买入、卖出建议。
3. 区分“事实事件”和“推断影响”。
4. 如果新闻主要是龙虎榜、高换手、异动快讯，要重点分析短线资金博弈和波动风险。
5. 如果出现题材标签，要判断它是明确题材催化，还是只是弱相关信息。
6. 输出要适合后续交易员 Agent 和风控 Agent 阅读。

输出格式：

## 新闻面结论
用 2-4 句话总结新闻面状态。

## 关键事件
列出最值得关注的新闻事件和对应事实。

## 可能影响
说明这些事件可能带来的短线情绪、资金行为或风险。

## 风险提示
说明有哪些信息不足、噪音、误判可能。

## 给后续 Agent 的提示
告诉后续交易员 Agent、风控 Agent 应该重点检查什么。

股票代码：
{symbol}

新闻事件信号：
{events_text}
"""


def run_news_agent(
    symbol: str,
    news_events: list[event_extractor.NewsEvent],
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> NewsAgentResult:
    """运行新闻事件分析 Agent。

    参数说明：
    - symbol：股票代码，比如 002361。
    - news_events：第 12 步抽取出的新闻事件列表。
    - provider：大模型平台，默认 deepseek。
    - model：具体模型名，不传则使用第 08 步默认模型。
    - temperature：模型随机性，新闻分析也建议低一点。
    """
    events_text = event_extractor.render_news_events_text(news_events)
    prompt = build_news_agent_prompt(symbol, events_text)

    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return NewsAgentResult(
        symbol=symbol,
        provider=response.provider,
        model=response.model,
        events_text=events_text,
        prompt=prompt,
        report_text=response.text,
    )


def render_news_agent_result(result: NewsAgentResult) -> str:
    """把新闻 Agent 的结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 新闻事件信号 ========
{result.events_text}

======== 新闻 Agent 报告 ========
{result.report_text}
"""


def demo_news_agent() -> None:
    """演示新闻 Agent。

    默认测试股票：
    - 002361 神剑股份

    选择它是因为它最近新闻里包含龙虎榜、高换手、涨停/大跌、题材异动，
    比较适合测试新闻事件分析。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    news_items = stock_news.get_stock_news(symbol, max_items=max_items)
    events = event_extractor.extract_news_events(news_items)
    result = run_news_agent(
        symbol=symbol,
        news_events=events,
        provider=provider,
        model=model,
    )
    print(render_news_agent_result(result))


if __name__ == "__main__":
    demo_news_agent()
