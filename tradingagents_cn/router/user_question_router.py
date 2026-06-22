"""用户自然语言问题路由。

这个文件解决的问题：

    用户不会直接输入：
        run_research_report_pipeline("000725")

    用户会问：
        帮我看看京东方A能不能买
        大唐发电行情怎么样
        给我看看今天的股市情况
        帮我筛几个短线机会

所以这里要把“人的问题”转换成“程序能执行的结构化请求”。

当前原则：
    只要问题里识别到单只股票，
    默认就走完整 TradingAgents 分析链路。

也就是说：
    “大唐发电行情怎么样”

不会只看实时行情，
而是识别成：

    single_stock_analysis

后续会跑：
    Market / News / Fundamentals / Summary / Debate / Risk / Portfolio。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tradingagents_cn.dataflows.stock_directory import (
    StockSymbolMatch,
    resolve_stocks_from_text,
)


class UserQuestionIntent(str, Enum):
    """用户问题意图。"""

    SINGLE_STOCK_ANALYSIS = "single_stock_analysis"
    MULTI_STOCK_ANALYSIS = "multi_stock_analysis"
    MARKET_OVERVIEW = "market_overview"
    STOCK_SCREENING = "stock_screening"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StockRouteItem:
    """多股问题中的单只股票。"""

    symbol: str
    stock_name: str
    match_source: str = ""


@dataclass
class UserQuestionRoute:
    """用户问题路由结果。

    intent:
        问题意图。

    original_question:
        用户原始问题。

    symbol:
        如果识别到单只股票，这里是 6 位股票代码。

    stock_name:
        如果识别到单只股票，这里是股票名称。

    stock_items:
        如果识别到多只股票，这里按用户提问顺序保存股票列表。

    analysis_depth:
        分析深度。
        当前单股问题默认 full。

    reason:
        为什么路由到这个意图。
        这个字段方便你调试。
    """

    intent: UserQuestionIntent
    original_question: str
    symbol: str | None = None
    stock_name: str | None = None
    stock_items: tuple[StockRouteItem, ...] = ()
    analysis_depth: str = "full"
    reason: str = ""


def route_user_question(question: str) -> UserQuestionRoute:
    """把用户自然语言问题转换成结构化路由。

    当前规则：

        1. 如果能识别出股票名称或 6 位代码：
           -> single_stock_analysis

        2. 如果没有股票，但包含“大盘 / 股市 / 今天市场”等词：
           -> market_overview

        3. 如果包含“筛股 / 推荐股票 / 选股”等词：
           -> stock_screening

        4. 其他：
           -> unknown
    """
    original_question = str(question or "").strip()
    if not original_question:
        return UserQuestionRoute(
            intent=UserQuestionIntent.UNKNOWN,
            original_question=original_question,
            reason="用户问题为空。",
        )

    stock_matches = resolve_stocks_from_text(original_question, use_akshare=False)
    if len(stock_matches) >= 2:
        return build_multi_stock_route(original_question, stock_matches)
    if len(stock_matches) == 1:
        return build_single_stock_route(original_question, stock_matches[0])

    if is_stock_screening_question(original_question):
        return UserQuestionRoute(
            intent=UserQuestionIntent.STOCK_SCREENING,
            original_question=original_question,
            analysis_depth="full",
            reason="问题没有识别到单只股票，但包含筛股/推荐/选股类关键词。",
        )

    if is_market_overview_question(original_question):
        return UserQuestionRoute(
            intent=UserQuestionIntent.MARKET_OVERVIEW,
            original_question=original_question,
            analysis_depth="full",
            reason="问题没有识别到单只股票，但包含大盘/股市/市场概览类关键词。",
        )

    stock_matches = resolve_stocks_from_text(original_question, use_akshare=True)
    if len(stock_matches) >= 2:
        return build_multi_stock_route(original_question, stock_matches)
    if len(stock_matches) == 1:
        return build_single_stock_route(original_question, stock_matches[0])

    return UserQuestionRoute(
        intent=UserQuestionIntent.UNKNOWN,
        original_question=original_question,
        reason="没有识别到股票，也没有命中已支持的问题类型。",
    )


def build_single_stock_route(
    question: str,
    stock_match: StockSymbolMatch,
) -> UserQuestionRoute:
    """构造单股完整分析路由。

    注意：
        这里故意不再区分“行情怎么样”“能不能买”。

        只要是单只股票问题，
        默认都走完整链路。
    """
    return UserQuestionRoute(
        intent=UserQuestionIntent.SINGLE_STOCK_ANALYSIS,
        original_question=question,
        symbol=stock_match.symbol,
        stock_name=stock_match.stock_name,
        analysis_depth="full",
        reason=(
            "问题中识别到单只股票，按当前产品原则默认执行完整 TradingAgents 分析。"
            f"匹配来源：{stock_match.match_source}。"
        ),
    )


def build_multi_stock_route(
    question: str,
    stock_matches: list[StockSymbolMatch],
) -> UserQuestionRoute:
    """构造多股完整分析路由。

    多股问题不是“选股推荐”。
    用户已经明确给出几只股票，所以程序应该逐只运行完整单股链路，
    最后再把每只股票的结论汇总给用户。
    """
    stock_items = tuple(
        StockRouteItem(
            symbol=stock_match.symbol,
            stock_name=stock_match.stock_name,
            match_source=stock_match.match_source,
        )
        for stock_match in stock_matches
    )
    stock_text = "、".join(f"{item.stock_name}（{item.symbol}）" for item in stock_items)
    return UserQuestionRoute(
        intent=UserQuestionIntent.MULTI_STOCK_ANALYSIS,
        original_question=question,
        stock_items=stock_items,
        analysis_depth="full",
        reason=f"问题中识别到多只股票，逐只执行完整分析：{stock_text}。",
    )


def is_market_overview_question(question: str) -> bool:
    """判断是否是市场概览问题。"""
    keywords = [
        "股市情况",
        "市场情况",
        "今天股市",
        "今日股市",
        "大盘",
        "行情整体",
        "市场概览",
        "今天市场",
        "今日市场",
    ]
    return any(keyword in question for keyword in keywords)


def is_stock_screening_question(question: str) -> bool:
    """判断是否是选股/筛股问题。"""
    keywords = [
        "筛股",
        "选股",
        "推荐股票",
        "推荐几只",
        "短线机会",
        "可以买的股票",
        "适合买",
        "股票池",
    ]
    return any(keyword in question for keyword in keywords)
