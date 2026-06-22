"""无人值守自动模拟交易模块。"""

from tradingagents_cn.auto_trader.daemon import (
    AutoPaperTrader,
    AutoTraderCandidate,
    AutoTraderConfig,
    AutoTraderCycleResult,
    AutoTraderRunItem,
    build_candidate_from_screening_row,
    discover_screening_candidates,
    is_a_share_trading_time,
    load_watchlist_candidates,
)

__all__ = [
    "AutoPaperTrader",
    "AutoTraderCandidate",
    "AutoTraderConfig",
    "AutoTraderCycleResult",
    "AutoTraderRunItem",
    "build_candidate_from_screening_row",
    "discover_screening_candidates",
    "is_a_share_trading_time",
    "load_watchlist_candidates",
]
