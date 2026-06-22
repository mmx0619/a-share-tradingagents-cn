"""A 股数据质量校验工具。

这个文件不获取数据，也不调用大模型。

它只检查已经拿到的数据是否“够用、够新、够完整”。

为什么需要这一层？
    股票分析里，数据质量会直接影响结论。
    如果历史行情只有几行，技术指标就不可靠；
    如果新闻很旧，News Agent 就不能把它当成实时消息；
    如果实时行情缺少最新价，报告里就应该明确提示。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def validate_daily_history(
    history: pd.DataFrame,
    trade_date: str,
    min_rows: int = 120,
    stale_days: int = 10,
) -> list[str]:
    """校验历史日线行情质量。

    min_rows:
        最少需要多少个交易日。
        原版指标里有 200 日均线，但 A 股新股或停牌股可能不足。
        这里先用 120 行作为警戒线。

    stale_days:
        最新行情距离分析日期超过多少自然日时，提示数据可能过旧。
    """
    issues: list[str] = []
    if history is None or history.empty:
        return ["历史行情数据为空，无法进行可靠技术分析。"]

    if len(history) < min_rows:
        issues.append(
            f"历史行情数据较少：仅 {len(history)} 个交易日，"
            f"少于建议值 {min_rows}，中长期技术指标可靠性会下降。"
        )

    if "Date" not in history.columns:
        issues.append("历史行情缺少 Date 字段，无法判断数据新鲜度。")
        return issues

    dates = pd.to_datetime(history["Date"], errors="coerce").dropna()
    if dates.empty:
        issues.append("历史行情 Date 字段无法解析，无法判断数据新鲜度。")
        return issues

    latest_date = dates.max().date()
    analysis_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    gap_days = (analysis_date - latest_date).days
    if gap_days > stale_days:
        issues.append(
            f"历史行情最新日期为 {latest_date}，距离分析日期 {analysis_date} "
            f"已有 {gap_days} 天，可能存在停牌、数据延迟或日期选择问题。"
        )

    future_rows = dates[dates.dt.date > analysis_date]
    if not future_rows.empty:
        issues.append("历史行情包含分析日期之后的数据，存在未来函数风险。")

    return issues


def validate_realtime_quote(quote: Any) -> list[str]:
    """校验实时/近实时行情快照质量。"""
    issues: list[str] = []
    if quote is None:
        return ["实时行情为空，报告将缺少当前盘口快照。"]

    if getattr(quote, "latest_price", None) is None:
        issues.append("实时行情缺少最新价。")

    if getattr(quote, "previous_close", None) in {None, 0}:
        issues.append("实时行情缺少有效昨收价，涨跌幅参考可能不完整。")

    if getattr(quote, "amount", None) is None and getattr(quote, "volume", None) is None:
        issues.append("实时行情缺少成交量和成交额，无法判断当前交易活跃度。")

    if not getattr(quote, "update_time", None):
        issues.append("实时行情缺少更新时间，无法判断快照新鲜度。")

    return issues


def validate_stock_news_items(
    news_items: list[Any],
    trade_date: str,
    stale_days: int = 14,
) -> list[str]:
    """校验个股新闻质量。

    如果最近新闻距离分析日期太久，
    News Agent 应该知道这些材料不是今日实时消息。
    """
    if not news_items:
        return ["个股新闻为空，News Agent 将缺少消息面材料。"]

    publish_dates = []
    for item in news_items:
        parsed = parse_datetime_safely(getattr(item, "publish_time", ""))
        if parsed is not None:
            publish_dates.append(parsed.date())

    if not publish_dates:
        return ["个股新闻发布时间无法解析，无法判断新闻新鲜度。"]

    latest_news_date = max(publish_dates)
    analysis_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    gap_days = (analysis_date - latest_news_date).days
    if gap_days > stale_days:
        return [
            f"最近一条可解析新闻日期为 {latest_news_date}，"
            f"距离分析日期 {analysis_date} 已有 {gap_days} 天，"
            "新闻材料可能偏旧。"
        ]

    return []


def validate_announcement_items(
    announcement_items: list[Any],
    trade_date: str,
    stale_days: int = 180,
) -> list[str]:
    """校验公司公告材料质量。

    公告不像新闻那样每天都有。
    所以默认 180 天才提示偏旧，重点发现“完全没有公告”或“日期无法解析”。
    """
    if not announcement_items:
        return ["公司公告为空，News Agent 无法核实上市公司正式披露材料。"]

    publish_dates = []
    for item in announcement_items:
        parsed = parse_datetime_safely(getattr(item, "publish_time", ""))
        if parsed is not None:
            publish_dates.append(parsed.date())

    if not publish_dates:
        return ["公司公告发布时间无法解析，无法判断公告材料新鲜度。"]

    latest_date = max(publish_dates)
    analysis_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    gap_days = (analysis_date - latest_date).days
    if gap_days > stale_days:
        return [
            f"最近一条可解析公告日期为 {latest_date}，"
            f"距离分析日期 {analysis_date} 已有 {gap_days} 天，"
            "公告材料可能偏旧。"
        ]

    return []


def validate_sentiment_items(sentiment_items: list[Any]) -> list[str]:
    """校验情绪面材料质量。"""
    if not sentiment_items:
        return ["情绪面材料为空，Sentiment Agent 将缺少社区讨论样本。"]

    missing_title_count = 0
    missing_source_count = 0
    for item in sentiment_items:
        if not str(getattr(item, "title", "") or "").strip():
            missing_title_count += 1
        if not str(getattr(item, "source", "") or "").strip():
            missing_source_count += 1

    issues: list[str] = []
    if missing_title_count:
        issues.append(f"情绪面材料中有 {missing_title_count} 条缺少标题。")
    if missing_source_count:
        issues.append(f"情绪面材料中有 {missing_source_count} 条缺少来源。")
    return issues


def validate_fundamental_texts(materials: dict[str, str | None]) -> list[str]:
    """校验基本面文本材料是否缺失或过短。

    财务表字段非常多，进入 Prompt 前已经做过关键字段筛选。
    这里不判断财务好坏，只检查材料本身是否足够可读。
    """
    issues: list[str] = []
    for label, text in materials.items():
        content = str(text or "").strip()
        if not content:
            issues.append(f"{label}为空，Fundamentals Agent 将缺少该部分材料。")
        elif len(content) < 80:
            issues.append(f"{label}文本较短，可能只包含错误信息或极少字段。")
    return issues


def render_data_quality_issues(issues: list[str]) -> str:
    """把数据质量问题渲染成工具输出文本。

    工具返回给大模型后，模型就能知道：
        哪些数据可信度较低；
        哪些材料缺失；
        哪些结论需要保守处理。
    """
    if not issues:
        return "数据质量提示：未发现明显数据质量问题。"

    lines = ["数据质量提示："]
    lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines)


def parse_datetime_safely(text: str) -> datetime | None:
    """尽量解析常见新闻时间格式。"""
    value = str(text or "").strip()
    if not value:
        return None

    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]
    for pattern in candidates:
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None

    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()
