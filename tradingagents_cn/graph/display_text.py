"""面向用户的中文展示文本工具。

Agent 内部为了结构化校验，会保留英文枚举：
    Buy / Hold / Sell / Underweight

但最终终端和报告是给人读的，应该尽量用中文。
"""

from __future__ import annotations

import re


TRADER_ACTION_LABELS = {
    "Buy": "买入",
    "Hold": "持有或观望",
    "Sell": "卖出或回避",
    "BUY": "买入",
    "HOLD": "持有或观望",
    "SELL": "卖出或回避",
}


PORTFOLIO_RATING_LABELS = {
    "Buy": "买入",
    "Overweight": "增配，偏积极",
    "Hold": "持有或观望",
    "Underweight": "低配，偏谨慎",
    "Sell": "卖出或回避",
}


ANALYST_LABELS = {
    "market": "技术面分析员",
    "sentiment": "情绪面分析员",
    "news": "新闻公告分析员",
    "fundamentals": "基本面分析员",
}


ROLE_LABELS = {
    "Trader": "交易员",
    "Trader Agent": "交易员",
    "Research Manager": "研究经理",
    "Portfolio Manager": "组合经理",
    "Market Agent": "技术面分析员",
    "Sentiment Agent": "情绪面分析员",
    "News Agent": "新闻公告分析员",
    "Fundamentals Agent": "基本面分析员",
    "Summary Agent": "综合汇总员",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Aggressive Risk Analyst": "激进风险分析员",
    "Conservative Risk Analyst": "保守风险分析员",
    "Neutral Risk Analyst": "中性风险分析员",
}


PAPER_STATUS_LABELS = {
    "filled": "已成交",
    "skipped": "未成交",
    "price_unavailable": "无法获取成交价",
    "error": "执行异常",
    "pending": "等待复盘",
    "not_required": "无需复盘",
    "unknown": "未知状态",
}


RISK_ROLE_LABELS = {
    "aggressive": "激进风险分析员",
    "conservative": "保守风险分析员",
    "neutral": "中性风险分析员",
}


RISK_LEVEL_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}


DEBATE_ROLE_LABELS = {
    "bull": "多头研究员",
    "bear": "空头研究员",
}


STANCE_STRENGTH_LABELS = {
    "weak": "弱",
    "medium": "中等",
    "strong": "强",
}


RISK_BAND_LABELS = {
    "normal": "正常执行",
    "controlled": "限制仓位",
    "defensive": "防守试探",
    "blocked": "禁止新增仓位",
}


REQUIRED_ACTION_LABELS = {
    "allow_planned_buy": "允许按计划分批买入",
    "limit_position": "限制仓位",
    "small_probe_only": "只允许小仓位试探",
    "block_new_buy": "禁止新增买入",
    "reduce_or_exit": "减仓或退出",
    "observe_only": "只观察",
}


def translate_trader_action(action: str) -> str:
    """把交易动作翻译成中文。"""
    return TRADER_ACTION_LABELS.get(str(action or "").strip(), "未知动作")


def translate_portfolio_rating(rating: str) -> str:
    """把五档评级翻译成中文。"""
    return PORTFOLIO_RATING_LABELS.get(str(rating or "").strip(), "未知评级")


def translate_machine_action(action: str) -> str:
    """把机器交易信号翻译成中文。"""
    return TRADER_ACTION_LABELS.get(str(action or "").strip(), "未知信号")


def translate_role_name(role_name: str) -> str:
    """把 Agent 英文角色名翻译成中文。"""
    return ROLE_LABELS.get(str(role_name or "").strip(), str(role_name or "").strip())


def translate_paper_status(status: str) -> str:
    """把模拟盘英文状态翻译成中文。"""
    return PAPER_STATUS_LABELS.get(str(status or "").strip(), "未知状态")


def render_selected_analysts_cn(selected_analysts: tuple[str, ...]) -> str:
    """把启用的分析员列表渲染成中文。"""
    if not selected_analysts:
        return "未记录"
    return "、".join(ANALYST_LABELS.get(item, item) for item in selected_analysts)


def localize_report_text(text: str) -> str:
    """对最终 Markdown 中常见英文标签做一次中文替换。

    这不是模型翻译器，只处理项目自己生成的固定英文标签。
    """
    replacements = {
        "**Action**": "**交易动作**",
        "**Reasoning**": "**理由**",
        "**Entry Price**": "**入场参考价**",
        "**Stop Loss**": "**止损价**",
        "**Position Sizing**": "**仓位建议**",
        "FINAL TRANSACTION PROPOSAL": "最终交易提案",
        "**Rating**": "**评级**",
        "**Executive Summary**": "**核心摘要**",
        "**Investment Thesis**": "**投资逻辑**",
        "**Price Target**": "**目标价**",
        "**Time Horizon**": "**观察周期**",
        "**Recommendation**": "**研究评级**",
        "**Rationale**": "**研究理由**",
        "**Strategic Actions**": "**策略动作**",
        "**Risk Role**": "**风险角色**",
        "**Risk Level**": "**风险等级**",
        "**Allow Trade**": "**是否允许交易**",
        "**Key Risks**": "**关键风险**",
        "**Risk Triggers**": "**风险触发条件**",
        "**Mitigation Plan**": "**风险缓释方案**",
        "**Position Sizing Advice**": "**仓位建议**",
        "**Debate Argument**": "**辩论观点**",
        "**Debate Role**": "**多空角色**",
        "**Stance Strength**": "**观点强度**",
        "**Thesis**": "**核心论点**",
        "**Supporting Evidence**": "**支持证据**",
        "**Opponent Rebuttals**": "**对方观点反驳**",
        "**Uncertainties**": "**不确定性**",
        "**Investment Implication**": "**投资含义**",
        "**Risk Band**": "**风控带**",
        "**Required Action**": "**必要动作**",
        "**Allow New Position**": "**是否允许新开仓**",
        "**Allow Add Position**": "**是否允许加仓**",
        "**Max Position Pct**": "**最大目标仓位**",
        "**Max Single Add Pct**": "**单次最大加仓**",
        "**Summary**": "**摘要**",
        "**Reasons**": "**原因**",
        "**Constraints**": "**约束条件**",
        "Portfolio Manager": "组合经理",
        "Research Manager": "研究经理",
        "Trader Agent": "交易员",
        "Trader": "交易员",
        "Market Agent": "技术面分析员",
        "Sentiment Agent": "情绪面分析员",
        "News Agent": "新闻公告分析员",
        "Fundamentals Agent": "基本面分析员",
        "Summary Agent": "综合汇总员",
        "Bull Researcher": "多头研究员",
        "Bear Researcher": "空头研究员",
        "Aggressive Risk Analyst": "激进风险分析员",
        "Conservative Risk Analyst": "保守风险分析员",
        "Neutral Risk Analyst": "中性风险分析员",
        "Risk Control": "风险控制",
        "FINAL": "最终",
        "TRANSACTION": "交易",
        "PROPOSAL": "提案",
    }
    enum_replacements = {
        "BUY": "买入",
        "HOLD": "持有或观望",
        "SELL": "卖出或回避",
        "Buy": "买入",
        "Hold": "持有或观望",
        "Sell": "卖出或回避",
        "Overweight": "增配，偏积极",
        "Underweight": "低配，偏谨慎",
    }
    technical_term_replacements = {
        "EMA": "指数移动平均线",
        "SMA": "简单移动平均线",
        "MA": "移动平均线",
        "MACD": "指数平滑异同移动平均线",
        "RSI": "相对强弱指标",
        "MFI": "资金流量指标",
        "BOLL": "布林带",
        "KDJ": "随机指标",
        "PE": "市盈率",
        "PB": "市净率",
        "ROE": "净资产收益率",
    }
    localized = str(text or "")
    for source, target in replacements.items():
        localized = localized.replace(source, target)
    for source, target in enum_replacements.items():
        localized = re.sub(
            rf"(?<![A-Za-z]){re.escape(source)}(?![A-Za-z])",
            target,
            localized,
        )
    for source, target in technical_term_replacements.items():
        localized = re.sub(
            rf"(?<![A-Za-z]){re.escape(source)}(?![A-Za-z])",
            target,
            localized,
        )
    localized = localize_markdown_label_colons(localized)
    localized = localize_field_values(localized)
    localized = localize_round_text(localized)
    return localized


def localize_markdown_label_colons(text: str) -> str:
    """把中文 Markdown 标签后的英文冒号改成中文冒号。"""
    return re.sub(r"(\*\*[^*\n]*[\u4e00-\u9fff][^*\n]*\*\*):", r"\1：", text)


def localize_field_values(text: str) -> str:
    """把结构化字段里的英文枚举值翻译成中文。

    只处理固定字段后面的枚举值，
    避免把普通英文技术缩写或正文误替换。
    """
    localized = text
    localized = replace_markdown_field_value(localized, "风险角色", RISK_ROLE_LABELS)
    localized = replace_markdown_field_value(localized, "风险等级", RISK_LEVEL_LABELS)
    localized = replace_markdown_field_value(localized, "多空角色", DEBATE_ROLE_LABELS)
    localized = replace_markdown_field_value(localized, "观点强度", STANCE_STRENGTH_LABELS)
    localized = replace_markdown_field_value(localized, "风控带", RISK_BAND_LABELS)
    localized = replace_markdown_field_value(localized, "必要动作", REQUIRED_ACTION_LABELS)
    return localized


def replace_markdown_field_value(text: str, label: str, mapping: dict[str, str]) -> str:
    """替换形如 `**字段**：value` 的固定枚举值。"""
    localized = text
    for source, target in mapping.items():
        localized = re.sub(
            rf"(\*\*{re.escape(label)}\*\*\s*[：:]\s*){re.escape(source)}(?=\s|$|。|，|,)",
            rf"\1{target}",
            localized,
        )
    return localized


def localize_round_text(text: str) -> str:
    """把 `Round 1` 这类轮次文本翻译成中文。"""
    localized = re.sub(r"\bRound\s+(\d+)\b", r"第 \1 轮", text)
    return re.sub(r"(第\s+\d+\s+轮):", r"\1：", localized)
