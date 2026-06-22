"""A 股综合研究 Agent。

这个 Agent 对应 TradingAgents 里“研究管理 / 综合汇总”的角色。

它不直接抓取新数据。
它的输入来自前面多个分析 Agent：

    Market Agent 报告
        技术面、行情、指标。

    News Agent 报告
        新闻、公告、消息面、宏观材料。

    Sentiment Agent 报告
        社区情绪、题材热度、分歧和反身性风险。

    Fundamentals Agent 报告
        公司资料、财务报表、基本面材料。

它的任务是：

    把多份报告放在一起，
    找出一致结论、冲突点、缺失信息和后续需要核实的问题，
    形成一份综合研究结论。

注意：
    这里仍然不是最终交易员。
    不直接输出“买入/卖出/持有”的交易动作。
    后续会再交给 Risk Agent、Trader Agent 等节点。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


SUMMARY_ANALYST_SYSTEM_PROMPT = """你是 A 股综合研究经理，任务是汇总多个分析员的报告。

你会收到多份材料：

1. Market Agent 技术面报告。
2. Sentiment Agent 情绪面报告。
3. News Agent 新闻面报告。
4. Fundamentals Agent 基本面报告。

你的目标不是重复三份报告，
而是做综合判断：

1. 提炼多份报告共同指向的核心结论。
2. 找出多份报告之间的矛盾或张力。
3. 明确哪些结论证据较强，哪些结论证据不足。
4. 标出仍需继续核实的数据缺口。
5. 给后续 Risk Agent 和 Trader Agent 提供清晰上下文。

输出要求：

1. 必须基于输入报告，不要编造新行情、新新闻或新财务数据。
2. 如果某个维度材料缺失或质量较弱，要明确说明。
3. 不要直接给最终买入/卖出结论。
4. 使用 Markdown 输出。
5. 最后用 Markdown 表格整理：
   - 维度；
   - 主要发现；
   - 证据强度；
   - 主要风险；
   - 后续需要核实的问题。
6. 本报告用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class SummaryAgentContext:
    """Summary Agent 的输入上下文。

    symbol:
        6 位股票代码。

    trade_date:
        当前分析日期。

    market_report:
        Market Agent 生成的报告。

    news_report:
        News Agent 生成的报告。

    sentiment_report:
        Sentiment Agent 生成的报告。
        如果本次没有启用 Sentiment Agent，这里会放入明确的缺失说明。

    fundamentals_report:
        Fundamentals Agent 生成的报告。

    prompt:
        最终准备给大模型的完整提示词。
    """

    symbol: str
    trade_date: str
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    prompt: str


def build_summary_agent_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    sentiment_report: str = "",
) -> SummaryAgentContext:
    """构造 Summary Agent 上下文。

    这个函数只负责把前三个 Agent 的报告整理成 Prompt。
    它不调用大模型，也不修改前三份报告。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_summary_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        market_report=market_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
    )

    return SummaryAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        market_report=market_report,
        sentiment_report=sentiment_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        prompt=prompt,
    )


def build_summary_agent_prompt(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    sentiment_report: str = "",
) -> str:
    """构造给大模型的 Summary Agent Prompt。

    这里的重点是：
        前面的 Analyst Agent 已经各自做了分析；
        Summary Agent 不应该简单复制；
        它应该做交叉验证、冲突识别和结论压缩。
    """
    actual_sentiment_report = (
        sentiment_report.strip()
        if sentiment_report and sentiment_report.strip()
        else "本次未启用 Sentiment Agent，或没有生成独立情绪面报告。"
    )
    return f"""{SUMMARY_ANALYST_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是 Market Agent 技术面报告：

{market_report}

下面是 Sentiment Agent 情绪面报告：

{actual_sentiment_report}

下面是 News Agent 新闻面报告：

{news_report}

下面是 Fundamentals Agent 基本面报告：

{fundamentals_report}

请基于以上材料，撰写综合研究结论。
"""


def render_summary_agent_context(context: SummaryAgentContext) -> str:
    """渲染 Summary Agent 上下文，方便调试时阅读。"""
    return context.prompt
