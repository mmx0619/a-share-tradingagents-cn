"""A 股模拟盘自动交易模块。"""

from tradingagents_cn.paper_trading.simulator import (
    DEFAULT_PAPER_LEDGER_PATH,
    PaperAccount,
    PaperPosition,
    PaperTrade,
    PaperTradingConfig,
    build_account_summary,
    load_paper_account,
    review_pending_paper_trades,
    run_paper_trading_from_result,
    save_paper_account,
)

__all__ = [
    "DEFAULT_PAPER_LEDGER_PATH",
    "PaperAccount",
    "PaperPosition",
    "PaperTrade",
    "PaperTradingConfig",
    "build_account_summary",
    "load_paper_account",
    "review_pending_paper_trades",
    "run_paper_trading_from_result",
    "save_paper_account",
]
