"""A 股多空研究员。

这个文件对应原版 TradingAgents 里的：

    bull_researcher.py
    bear_researcher.py

原版的结构是：

    分析员报告完成
      -> Bull Researcher
      -> Bear Researcher
      -> Bull Researcher
      -> ...
      -> Research Manager

也就是说，Bull / Bear 不是重新抓数据的 Agent。
它们的工作是基于前面分析员的报告进行辩论：

    Bull Researcher:
        尽可能构建“看多/支持投资”的证据链。

    Bear Researcher:
        尽可能构建“看空/反对投资”的风险链。

注意：
    这里的“Bull / Bear”不是最终结论。
    它们是辩论角色。
    后面还需要 Research Manager 对辩论进行裁判和汇总。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


BULL_RESEARCHER_SYSTEM_PROMPT = """你是 A 股多头研究员，负责为投资该股票构建证据充分的正方论点。

你的任务是强调：

1. 成长潜力：
   关注公司市场机会、收入增长可能性、业务扩张空间、行业景气度。

2. 竞争优势：
   关注产品、品牌、渠道、成本、行业地位、商业模式等优势。

3. 正面指标：
   使用技术面、基本面、行业趋势、新闻事件中的正面证据。

4. 对空头观点的反驳：
   如果已经有空头观点，需要用具体材料和逻辑反驳，
   说明为什么多头视角更有说服力。

5. 辩论参与感：
   不要只是罗列事实，要像在和空头研究员辩论一样，
   直接回应对方观点。

输出要求：

1. 必须基于输入材料，不要编造新行情、新新闻或新财务数据。
2. 可以强调有利证据，但不能隐藏关键不确定性。
3. 不要直接给最终买入/卖出结论。
4. 输出必须是合法 JSON，不要输出 Markdown，不要加解释文字。
5. 本论点用于个人投资研究辅助，最终决策由使用者自行确认。
"""


BEAR_RESEARCHER_SYSTEM_PROMPT = """你是 A 股空头研究员，负责为反对投资该股票构建证据充分的反方论点。

你的任务是强调：

1. 风险和挑战：
   关注市场饱和、估值压力、财务不稳定、宏观风险、政策风险等因素。

2. 竞争弱点：
   关注市场地位削弱、创新不足、竞品威胁、行业格局恶化等因素。

3. 负面指标：
   使用技术面、基本面、行业趋势、新闻事件中的负面证据。

4. 对多头观点的反驳：
   如果已经有多头观点，需要用具体材料和逻辑指出其中的薄弱处，
   识别过度乐观或证据不足的假设。

5. 辩论参与感：
   不要只是罗列事实，要像在和多头研究员辩论一样，
   直接回应对方观点。

输出要求：

1. 必须基于输入材料，不要编造新行情、新新闻或新财务数据。
2. 可以强调风险证据，但不能故意忽略明显的正面材料。
3. 不要直接给最终买入/卖出结论。
4. 输出必须是合法 JSON，不要输出 Markdown，不要加解释文字。
5. 本论点用于个人投资研究辅助，最终决策由使用者自行确认。
"""


class DebateArgument(BaseModel):
    """多空研究员输出的结构化辩论观点。

    role:
        当前辩论角色，必须是 bull 或 bear。

    stance_strength:
        本轮立场强度。
        strong 表示证据链较强，medium 表示证据中等，weak 表示证据较弱。

    thesis:
        本轮辩论的核心主张。

    supporting_evidence:
        支撑本方观点的证据列表。

    opponent_rebuttals:
        对上一轮对手观点的反驳列表。

    uncertainties:
        本方也承认的不确定性，避免模型只说单边好话或坏话。

    investment_implication:
        对 Research Manager 有用的投资含义。
        这里不是最终买卖决策，而是说明本方观点对后续裁判的影响。

    debate_argument:
        可以直接放进辩论历史的完整论点文本。
    """

    role: Literal["bull", "bear"] = Field(description="多空辩论角色。")
    stance_strength: Literal["weak", "medium", "strong"] = Field(description="本轮立场强度。")
    thesis: str = Field(description="核心主张。")
    supporting_evidence: list[str] = Field(description="支撑本方观点的证据列表。")
    opponent_rebuttals: list[str] = Field(description="对对手观点的反驳列表。")
    uncertainties: list[str] = Field(description="本方观点承认的不确定性。")
    investment_implication: str = Field(description="对后续 Research Manager 的投资含义。")
    debate_argument: str = Field(description="完整辩论论点文本。")


DEBATE_ARGUMENT_JSON_REQUIREMENT = """输出要求：

你必须只输出一个合法 JSON 对象，不要输出 Markdown，不要加解释文字。

JSON 格式必须是：

{
  "role": "bull | bear",
  "stance_strength": "weak | medium | strong",
  "thesis": "本轮辩论的核心主张",
  "supporting_evidence": ["证据 1", "证据 2"],
  "opponent_rebuttals": ["对对手观点的反驳 1", "对对手观点的反驳 2"],
  "uncertainties": ["不确定性 1", "不确定性 2"],
  "investment_implication": "对 Research Manager 的投资含义，但不要直接给最终买卖决策",
  "debate_argument": "完整辩论论点文本"
}

字段要求：

1. role 必须等于当前辩论角色。
2. stance_strength 只能是 weak / medium / strong。
3. supporting_evidence、opponent_rebuttals、uncertainties 必须是字符串数组。
4. 不要编造输入材料里没有的具体价格、新闻、公告或财务数字。
5. 不要直接输出最终 Buy / Hold / Sell 决策。
"""


@dataclass
class DebateResearcherContext:
    """多空研究员上下文。

    symbol:
        6 位 A 股代码。

    trade_date:
        分析日期。

    role:
        当前辩论角色。
        取值通常是 bull 或 bear。

    prompt:
        最终发给大模型的完整 Prompt。
    """

    symbol: str
    trade_date: str
    role: str
    prompt: str


def build_bull_researcher_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    summary_report: str,
    debate_history: str = "",
    last_bear_argument: str = "",
) -> DebateResearcherContext:
    """构造 Bull Researcher 的上下文。

    这个函数对应原版 TradingAgents 的 Bull Researcher Prompt。

    原版会读取：
        Market research report
        Social media sentiment report
        Latest world affairs news
        Company fundamentals report
        Conversation history
        Last bear argument

    当前 A 股版暂时没有独立 sentiment_report，
    所以先明确写成“暂未接入独立情绪报告”。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_bull_researcher_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        summary_report=summary_report,
        debate_history=debate_history,
        last_bear_argument=last_bear_argument,
    )
    return DebateResearcherContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        role="bull",
        prompt=prompt,
    )


def build_bear_researcher_context(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    summary_report: str,
    debate_history: str = "",
    last_bull_argument: str = "",
) -> DebateResearcherContext:
    """构造 Bear Researcher 的上下文。

    这个函数对应原版 TradingAgents 的 Bear Researcher Prompt。

    Bear Researcher 会读取上一轮多头观点，
    然后基于同一批研究材料进行反驳。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    prompt = build_bear_researcher_prompt(
        symbol=normalized_symbol,
        trade_date=trade_date,
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        summary_report=summary_report,
        debate_history=debate_history,
        last_bull_argument=last_bull_argument,
    )
    return DebateResearcherContext(
        symbol=normalized_symbol,
        trade_date=trade_date,
        role="bear",
        prompt=prompt,
    )


def build_bull_researcher_prompt(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    summary_report: str,
    debate_history: str = "",
    last_bear_argument: str = "",
) -> str:
    """构造 Bull Researcher Prompt。"""
    return f"""{BULL_RESEARCHER_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

可用研究材料如下：

## Market Agent 技术面报告

{market_report}

## News Agent 新闻面报告

{news_report}

## 独立社媒情绪报告

当前 A 股版暂未接入独立情绪报告。
如果新闻报告中包含投资者情绪或舆情内容，可以谨慎引用；
否则不要编造社媒情绪。

## Fundamentals Agent 基本面报告

{fundamentals_report}

## Summary Agent 综合研究结论

{summary_report}

## 当前辩论历史

{debate_history or "暂无辩论历史。"}

## 上一轮空头观点

{last_bear_argument or "暂无空头观点。"}

请基于以上材料，输出一段有说服力的多头论点，并针对空头担忧进行回应。

{DEBATE_ARGUMENT_JSON_REQUIREMENT}
"""


def build_bear_researcher_prompt(
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    summary_report: str,
    debate_history: str = "",
    last_bull_argument: str = "",
) -> str:
    """构造 Bear Researcher Prompt。"""
    return f"""{BEAR_RESEARCHER_SYSTEM_PROMPT}

当前分析股票：{symbol}
当前分析日期：{trade_date}

可用研究材料如下：

## Market Agent 技术面报告

{market_report}

## News Agent 新闻面报告

{news_report}

## 独立社媒情绪报告

当前 A 股版暂未接入独立情绪报告。
如果新闻报告中包含投资者情绪或舆情内容，可以谨慎引用；
否则不要编造社媒情绪。

## Fundamentals Agent 基本面报告

{fundamentals_report}

## Summary Agent 综合研究结论

{summary_report}

## 当前辩论历史

{debate_history or "暂无辩论历史。"}

## 上一轮多头观点

{last_bull_argument or "暂无多头观点。"}

请基于以上材料，输出一段有说服力的空头论点，并针对多头观点进行回应。

{DEBATE_ARGUMENT_JSON_REQUIREMENT}
"""


def render_debate_researcher_context(context: DebateResearcherContext) -> str:
    """渲染多空研究员上下文，方便调试时阅读。"""
    return context.prompt


def render_debate_argument(argument: DebateArgument) -> str:
    """把结构化多空论点渲染成 Research Manager 可读文本。"""
    lines = [
        f"**Debate Role**: {argument.role}",
        f"**Stance Strength**: {argument.stance_strength}",
        "",
        f"**Thesis**: {argument.thesis}",
        "",
        "**Supporting Evidence**:",
        *[f"- {item}" for item in argument.supporting_evidence],
        "",
        "**Opponent Rebuttals**:",
        *[f"- {item}" for item in argument.opponent_rebuttals],
        "",
        "**Uncertainties**:",
        *[f"- {item}" for item in argument.uncertainties],
        "",
        f"**Investment Implication**: {argument.investment_implication}",
        "",
        f"**Debate Argument**: {argument.debate_argument}",
    ]
    return "\n".join(lines)


def build_fallback_debate_argument(role: str, error_message: str) -> DebateArgument:
    """构造多空辩论员的保守兜底输出。"""
    normalized_role = role if role in {"bull", "bear"} else "bear"
    return DebateArgument(
        role=normalized_role,  # type: ignore[arg-type]
        stance_strength="weak",
        thesis="模型没有返回可校验的多空辩论结构，本轮观点只能作为低置信度材料。",
        supporting_evidence=[
            "结构化辩论输出失败，无法确认本方证据链。",
        ],
        opponent_rebuttals=[
            "由于输出校验失败，本轮无法可靠反驳对手观点。",
        ],
        uncertainties=[
            "模型输出没有通过程序校验。",
            f"结构化输出错误：{error_message}",
        ],
        investment_implication="Research Manager 应降低本轮辩论材料权重，并优先依赖已校验的分析报告。",
        debate_argument="本轮多空辩论结构化输出失败，采用低置信度保守兜底观点。",
    )


def debate_argument_to_json_schema() -> dict:
    """返回 DebateArgument JSON Schema。"""
    return DebateArgument.model_json_schema()
