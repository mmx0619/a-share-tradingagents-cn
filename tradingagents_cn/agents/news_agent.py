"""A 股新闻分析 Agent。

这个文件按照原版 TradingAgents 的 News Analyst 思路来写。

原版 News Analyst 的核心不是在代码里写死“某类新闻=利好/利空”，
而是：

    1. 给大模型新闻工具；
    2. 让大模型读取公司相关新闻和宏观新闻；
    3. 要求大模型写出对交易和宏观环境有帮助的新闻报告；
    4. 报告最后附 Markdown 表格整理关键点。

当前 A 股版先做“新闻上下文和提示词准备”。
真正的大模型调用会在后续图工作流里完成。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.stock_news import (
    StockNewsItem,
    render_stock_news_text,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


NEWS_ANALYST_SYSTEM_PROMPT = """你是 A 股新闻研究员，任务是分析最近新闻和市场趋势。

你需要关注：

1. 个股相关新闻：
   包括公司公告、经营事件、股价异动、龙虎榜、行业消息、监管信息等。

2. 更广泛的市场和宏观信息：
   包括政策变化、行业景气度、流动性、市场风险偏好、突发事件等。

分析要求：

1. 基于提供的新闻原文和来源进行分析，不要编造没有出现的新闻。
2. 区分事实、推测和不确定性。
3. 如果新闻内容较旧、缺少实质信息，必须明确指出它的参考价值有限。
4. 不要只根据标题下结论，要结合发布时间、来源和正文内容。
5. 输出应帮助后续研究员和交易员理解消息面影响。
6. 不要直接给最终买入/卖出结论。
7. 报告最后用 Markdown 表格整理关键新闻、可能影响和可信度。
8. 本报告用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class NewsAgentContext:
    """News Agent 的输入上下文。

    symbol:
        6 位股票代码。

    trade_date:
        当前分析日期。

    stock_news_text:
        个股新闻文本。

    macro_news_text:
        可选宏观新闻文本。
        当前可以为空，后续接入财联社、证券时报等宏观新闻源。

    prompt:
        最终准备给大模型的完整提示词。
    """

    symbol: str
    trade_date: str
    stock_news_text: str
    macro_news_text: str | None
    prompt: str


def build_news_agent_context(
    symbol: str,
    trade_date: str,
    stock_news_items: list[StockNewsItem],
    macro_news_text: str | None = None,
) -> NewsAgentContext:
    """构造 News Agent 上下文。

    这个函数只负责把新闻原材料整理成 Prompt。
    它不判断利好利空，也不调用大模型。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    stock_news_text = render_stock_news_text(stock_news_items)
    prompt = build_news_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        stock_news_text=stock_news_text,
        macro_news_text=macro_news_text,
    )

    return NewsAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        stock_news_text=stock_news_text,
        macro_news_text=macro_news_text,
        prompt=prompt,
    )


def build_news_agent_prompt(
    symbol: str,
    trade_date: str,
    stock_news_text: str,
    macro_news_text: str | None = None,
) -> str:
    """构造给大模型的 News Agent Prompt。

    这里借鉴原版 TradingAgents：

    - 聚焦最近新闻和趋势；
    - 同时考虑个股和宏观；
    - 输出详细新闻研究报告；
    - 最后用 Markdown 表格整理关键点。
    """
    macro_section = macro_news_text or "暂未提供宏观新闻材料。"

    return f"""{NEWS_ANALYST_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是个股相关新闻材料：

{stock_news_text}

下面是宏观或市场新闻材料：

{macro_section}

请基于以上新闻材料撰写 A 股新闻面分析报告。
"""


def render_news_agent_context(context: NewsAgentContext) -> str:
    """渲染 News Agent 上下文，方便调试时阅读。"""
    return context.prompt
