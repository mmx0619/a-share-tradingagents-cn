"""A 股基本面分析 Agent。

这个文件按照原版 TradingAgents 的 Fundamentals Analyst 思路来写。

原版 Fundamentals Analyst 的核心是：

    1. 给大模型基本面工具；
    2. 让模型读取公司概况、财务数据和历史财务表现；
    3. 使用资产负债表、现金流量表、利润表等材料；
    4. 写出完整的基本面研究报告；
    5. 报告最后附 Markdown 表格整理关键点。

当前 A 股版先做“基本面上下文和提示词准备”。

注意：
    这里不在代码里写死财务好坏判断规则。
    例如不写“负债率超过多少一定危险”。
    代码只整理材料和提示词，分析交给后续大模型。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


FUNDAMENTALS_ANALYST_SYSTEM_PROMPT = """你是 A 股基本面研究员，任务是分析公司的基本面信息。

你需要关注：

1. 公司基本资料：
   包括主营业务、行业地位、业务结构、重要风险提示等。

2. 财务报表：
   包括资产负债表、利润表、现金流量表。

3. 财务历史：
   关注收入、利润、现金流、资产负债结构等变化。

4. 信息来源：
   A 股场景下，后续数据源应优先来自交易所公告、巨潮资讯、上市公司定期报告等公开披露材料。

分析要求：

1. 基于提供的基本面材料进行分析，不要编造没有出现的财务数据。
2. 区分事实、推测和不确定性。
3. 如果材料缺失，必须明确指出缺失项，不要自行补全。
4. 不要直接给最终买入/卖出结论。
5. 输出应帮助后续研究员、风控 Agent 和交易员理解公司基本面。
6. 报告最后用 Markdown 表格整理关键财务点、可能影响和需要继续核实的问题。
7. 本报告用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class FundamentalsAgentContext:
    """Fundamentals Agent 的输入上下文。

    symbol:
        6 位股票代码。

    trade_date:
        当前分析日期。

    company_profile_text:
        公司概况材料。

    balance_sheet_text:
        资产负债表材料。

    cashflow_text:
        现金流量表材料。

    income_statement_text:
        利润表材料。

    prompt:
        最终准备给大模型的完整提示词。
    """

    symbol: str
    trade_date: str
    company_profile_text: str | None
    balance_sheet_text: str | None
    cashflow_text: str | None
    income_statement_text: str | None
    prompt: str


def build_fundamentals_agent_context(
    symbol: str,
    trade_date: str,
    company_profile_text: str | None = None,
    balance_sheet_text: str | None = None,
    cashflow_text: str | None = None,
    income_statement_text: str | None = None,
) -> FundamentalsAgentContext:
    """构造 Fundamentals Agent 上下文。

    这个函数只整理材料和 Prompt。
    它不访问财报网站，也不调用大模型。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_fundamentals_agent_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        company_profile_text=company_profile_text,
        balance_sheet_text=balance_sheet_text,
        cashflow_text=cashflow_text,
        income_statement_text=income_statement_text,
    )

    return FundamentalsAgentContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        company_profile_text=company_profile_text,
        balance_sheet_text=balance_sheet_text,
        cashflow_text=cashflow_text,
        income_statement_text=income_statement_text,
        prompt=prompt,
    )


def build_fundamentals_agent_prompt(
    symbol: str,
    trade_date: str,
    company_profile_text: str | None = None,
    balance_sheet_text: str | None = None,
    cashflow_text: str | None = None,
    income_statement_text: str | None = None,
) -> str:
    """构造给大模型的 Fundamentals Agent Prompt。

    这里借鉴原版 TradingAgents：

    - 分析公司基本面；
    - 使用公司资料和财务报表；
    - 尽量详细；
    - 最后用 Markdown 表格整理关键点。
    """
    return f"""{FUNDAMENTALS_ANALYST_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

下面是公司基本资料：

{company_profile_text or "暂未提供公司基本资料。"}

下面是资产负债表材料：

{balance_sheet_text or "暂未提供资产负债表材料。"}

下面是现金流量表材料：

{cashflow_text or "暂未提供现金流量表材料。"}

下面是利润表材料：

{income_statement_text or "暂未提供利润表材料。"}

请基于以上材料撰写 A 股基本面分析报告。
"""


def render_fundamentals_agent_context(context: FundamentalsAgentContext) -> str:
    """渲染 Fundamentals Agent 上下文，方便调试时阅读。"""
    return context.prompt
