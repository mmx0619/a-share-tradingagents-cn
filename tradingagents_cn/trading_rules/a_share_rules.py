"""A 股买入范围规则。

这个文件只处理“哪些股票允许进入自动买入范围”。

当前用户指定的策略边界是：

    只允许买沪深主板普通 A 股。

也就是日常涨跌幅通常为 10% 的主板普通股票。
创业板、科创板、北交所、ST、退市整理股、新股特殊阶段先全部排除。

注意：
    这里是程序硬规则，不交给大模型自由判断。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


MAIN_BOARD_COMMON_PREFIXES = (
    # 深市主板。002 原中小板已并入深市主板，这里按主板普通股票处理。
    "000",
    "001",
    "002",
    "003",
    # 沪市主板。
    "600",
    "601",
    "603",
    "605",
)

CHINEXT_PREFIXES = ("300", "301")
STAR_MARKET_PREFIXES = ("688", "689")
BSE_PREFIXES = ("4", "8", "920")


@dataclass(frozen=True)
class AShareBuyRuleDecision:
    """A 股买入范围判断结果。

    allowed_to_buy:
        是否允许自动模拟买入。

    board:
        程序识别出的板块类型。

    price_limit_pct:
        常规日涨跌幅限制比例。
        这里只用于说明和日志，不直接保证真实交易价格范围。

    reason:
        为什么允许或拒绝。
    """

    symbol: str
    name: str
    board: str
    allowed_to_buy: bool
    price_limit_pct: float | None
    reason: str


def evaluate_a_share_buy_universe(
    symbol: str,
    name: str | None = None,
    allowed_universe: str = "main_board_common",
) -> AShareBuyRuleDecision:
    """判断某只股票是否允许自动买入。

    当前唯一正式买入范围：
        main_board_common

    也就是沪深主板普通 A 股。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    stock_name = str(name or "").strip()

    if allowed_universe != "main_board_common":
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="unsupported_universe",
            allowed_to_buy=False,
            price_limit_pct=None,
            reason=f"当前不支持自动买入范围：{allowed_universe}。",
        )

    if is_risk_warning_or_delisting_name(stock_name):
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="risk_warning_or_delisting",
            allowed_to_buy=False,
            price_limit_pct=None,
            reason="股票名称包含 ST、*ST 或退市风险标记，不属于主板普通股票买入范围。",
        )

    if normalized_symbol.startswith(MAIN_BOARD_COMMON_PREFIXES):
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="main_board_common",
            allowed_to_buy=True,
            price_limit_pct=0.10,
            reason="属于沪深主板普通 A 股，符合当前只买 10% 涨跌幅主板普通股的策略边界。",
        )

    if normalized_symbol.startswith(CHINEXT_PREFIXES):
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="chinext",
            allowed_to_buy=False,
            price_limit_pct=0.20,
            reason="创业板股票不在当前自动买入范围内。",
        )

    if normalized_symbol.startswith(STAR_MARKET_PREFIXES):
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="star_market",
            allowed_to_buy=False,
            price_limit_pct=0.20,
            reason="科创板股票不在当前自动买入范围内。",
        )

    if normalized_symbol.startswith(BSE_PREFIXES):
        return AShareBuyRuleDecision(
            symbol=normalized_symbol,
            name=stock_name,
            board="bse",
            allowed_to_buy=False,
            price_limit_pct=0.30,
            reason="北交所股票不在当前自动买入范围内。",
        )

    return AShareBuyRuleDecision(
        symbol=normalized_symbol,
        name=stock_name,
        board="unknown",
        allowed_to_buy=False,
        price_limit_pct=None,
        reason="无法识别为沪深主板普通 A 股，不允许自动买入。",
    )


def is_main_board_common_stock(symbol: str, name: str | None = None) -> bool:
    """判断是否为当前允许自动买入的沪深主板普通 A 股。"""
    return evaluate_a_share_buy_universe(symbol, name).allowed_to_buy


def is_risk_warning_or_delisting_name(name: str | None) -> bool:
    """根据股票简称判断是否带风险警示或退市标记。"""
    text = str(name or "").strip().upper()
    if not text:
        return False

    risk_keywords = (
        "ST",
        "*ST",
        "退",
        "退市",
    )
    return any(keyword in text for keyword in risk_keywords)
