"""A 股 TradingAgents 自动模拟交易入口。

这个脚本用于无人值守运行：

    1. 定时扫描候选股；
    2. 对候选股运行完整 TradingAgents 分析；
    3. 根据最终 BUY / HOLD / SELL 和风控护栏写入模拟盘；
    4. 保存报告、full_state 和自动交易周期日志。

重要说明：
    当前只做模拟盘，不接券商，不真实下单。
"""

from __future__ import annotations

import argparse

from tradingagents_cn.auto_trader import AutoPaperTrader, AutoTraderConfig
from tradingagents_cn.dataflows.vendor_config import (
    default_data_vendors,
    default_tool_vendors,
    normalize_selected_analysts,
    normalize_vendor_overrides,
)
from tradingagents_cn.graph import ResearchInputConfig
from tradingagents_cn.llm.errors import LLMAPIError
from tradingagents_cn.llm.factory import create_chat_client


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="A 股 TradingAgents 自动模拟交易守护进程")

    parser.add_argument("--loop", action="store_true", help="长期循环运行；不传则只跑一轮")
    parser.add_argument("--max-cycles", type=int, default=None, help="长期运行时最多跑几轮；不传则不限")
    parser.add_argument("--scan-interval-seconds", type=int, default=300, help="两次扫描间隔秒数，默认 300")

    parser.add_argument("--watchlist", default=None, help="自选股文件，每行写 000725 或 000725,京东方A")
    parser.add_argument("--no-market-screening", action="store_true", help="关闭全市场行情快照筛选")
    parser.add_argument("--no-holdings", action="store_true", help="不自动复查模拟账户已有持仓")
    parser.add_argument("--max-candidates", type=int, default=3, help="每轮最多深度分析几只股票，默认 3")
    parser.add_argument("--candidate-pool-size", type=int, default=30, help="全市场初筛候选池大小，默认 30")

    parser.add_argument("--min-change-pct", type=float, default=3.0, help="涨幅触发线，默认 3")
    parser.add_argument("--max-drop-pct", type=float, default=-5.0, help="跌幅风险复查线，默认 -5")
    parser.add_argument("--min-turnover-rate", type=float, default=5.0, help="换手率触发线，默认 5")
    parser.add_argument("--min-volume-ratio", type=float, default=1.5, help="量比触发线，默认 1.5")

    parser.add_argument("--provider", default=None, choices=["deepseek", "openai", "kimi", "gemini"], help="模型服务商")
    parser.add_argument("--model", default=None, help="模型名")
    parser.add_argument("--temperature", type=float, default=0.2, help="模型随机性，默认 0.2")
    parser.add_argument("--debate-rounds", type=int, default=1, help="多空辩论轮数，默认 1")
    parser.add_argument("--risk-rounds", type=int, default=1, help="风险辩论轮数，默认 1")

    parser.add_argument("--history-days", type=int, default=420, help="历史行情自然日长度，默认 420")
    parser.add_argument("--news-max-items", type=int, default=2, help="新闻最多条数，默认 2")
    parser.add_argument("--sentiment-max-items", type=int, default=12, help="情绪材料最多条数，默认 12")
    parser.add_argument("--fundamentals-max-rows", type=int, default=2, help="财务表最多行数，默认 2")
    parser.add_argument("--fundamentals-max-columns", type=int, default=20, help="财务表最多列数，默认 20")
    parser.add_argument(
        "--analysts",
        default="market,sentiment,news,fundamentals",
        help="启用哪些 Analyst，逗号分隔",
    )
    parser.add_argument(
        "--sentiment-sources",
        default="eastmoney,xueqiu,tonghuashun,taoguba",
        help="情绪源，逗号分隔",
    )
    parser.add_argument("--no-realtime", action="store_true", help="关闭实时/近实时行情获取")
    parser.add_argument("--no-news", action="store_true", help="关闭新闻获取")
    parser.add_argument("--no-sentiment", action="store_true", help="关闭独立 Sentiment Analyst")
    parser.add_argument("--no-fundamentals", action="store_true", help="关闭基本面获取")

    parser.add_argument("--benchmark-symbol", default="000300", help="复盘基准指数代码，默认 000300")
    parser.add_argument("--benchmark-name", default="沪深300", help="复盘基准名称，默认 沪深300")
    parser.add_argument("--memory-holding-days", type=int, default=5, help="记忆复盘持有天数，默认 5")
    parser.add_argument("--resolve-all-pending-memory", action="store_true", help="尝试复盘所有 pending 记忆")

    parser.add_argument("--paper-ledger", default="outputs/paper_trading/account.json", help="模拟盘账户 JSON 路径")
    parser.add_argument("--paper-initial-cash", type=float, default=10000.0, help="模拟盘初始资金，默认 10000")
    parser.add_argument("--paper-max-position-pct", type=float, default=0.20, help="单只股票模拟仓位上限，默认 0.20")
    parser.add_argument("--paper-min-trade-amount", type=float, default=1000.0, help="模拟盘最小成交金额，默认 1000")
    parser.add_argument("--paper-review-days", type=int, default=5, help="模拟成交复盘天数，默认 5")
    parser.add_argument("--no-paper-trading", action="store_true", help="只自动分析，不写入模拟盘")
    parser.add_argument("--no-paper-review", action="store_true", help="不复盘历史 pending 模拟成交")
    parser.add_argument(
        "--allow-after-hours-paper-trading",
        action="store_true",
        help="非交易时段也允许按最近价格写入模拟成交",
    )

    parser.add_argument("--output-dir", default="outputs/auto_paper_trader", help="自动交易周期日志目录")
    parser.add_argument("--report-output-dir", default="outputs/user_questions", help="单股分析报告输出目录")
    parser.add_argument("--no-save-reports", action="store_true", help="不保存每只候选股的 Markdown 报告")
    parser.add_argument("--no-full-state", action="store_true", help="不保存 full_state.json")
    parser.add_argument("--full-state-dir", default="outputs/run_states", help="full_state.json 输出目录")
    parser.add_argument(
        "--data-vendor",
        action="append",
        default=[],
        help="覆盖数据源配置，格式 key=value",
    )
    parser.add_argument(
        "--tool-vendor",
        action="append",
        default=[],
        help="覆盖工具源配置，格式 key=value",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    return build_arg_parser().parse_args(argv)


def build_auto_config_from_args(args: argparse.Namespace) -> AutoTraderConfig:
    """把命令行参数转换成自动交易配置。"""
    return AutoTraderConfig(
        run_forever=args.loop,
        max_cycles=args.max_cycles,
        scan_interval_seconds=args.scan_interval_seconds,
        watchlist_path=args.watchlist,
        include_market_screening=not args.no_market_screening,
        include_holdings=not args.no_holdings,
        max_candidates_per_cycle=args.max_candidates,
        candidate_pool_size=args.candidate_pool_size,
        output_dir=args.output_dir,
        report_output_dir=args.report_output_dir,
        save_reports=not args.no_save_reports,
        execute_only_during_trading_hours=not args.allow_after_hours_paper_trading,
        min_change_pct=args.min_change_pct,
        max_drop_pct=args.max_drop_pct,
        min_turnover_rate=args.min_turnover_rate,
        min_volume_ratio=args.min_volume_ratio,
    )


def build_research_config_from_args(args: argparse.Namespace) -> ResearchInputConfig:
    """把命令行参数转换成研究链路配置。"""
    data_vendors = default_data_vendors()
    data_vendors.update(normalize_vendor_overrides(args.data_vendor))
    tool_vendors = default_tool_vendors()
    tool_vendors.update(normalize_vendor_overrides(args.tool_vendor))

    return ResearchInputConfig(
        history_calendar_days=args.history_days,
        news_max_items=args.news_max_items,
        sentiment_max_items=args.sentiment_max_items,
        fundamentals_max_rows=args.fundamentals_max_rows,
        fundamentals_max_columns=args.fundamentals_max_columns,
        include_realtime=not args.no_realtime,
        include_news=not args.no_news,
        include_sentiment=not args.no_sentiment,
        include_fundamentals=not args.no_fundamentals,
        selected_analysts=normalize_selected_analysts(args.analysts),
        sentiment_sources=args.sentiment_sources,
        data_vendors=data_vendors,
        tool_vendors=tool_vendors,
        save_full_state=not args.no_full_state,
        full_state_output_dir=args.full_state_dir,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_name=args.benchmark_name,
        memory_holding_days=args.memory_holding_days,
        resolve_all_pending_memory=args.resolve_all_pending_memory,
        enable_paper_trading=not args.no_paper_trading,
        paper_trading_ledger_path=args.paper_ledger,
        paper_trading_initial_cash=args.paper_initial_cash,
        paper_trading_max_single_position_pct=args.paper_max_position_pct,
        paper_trading_min_trade_amount=args.paper_min_trade_amount,
        paper_trading_review_pending=not args.no_paper_review,
        paper_trading_review_holding_days=args.paper_review_days,
    )


def build_llm_client_from_args(args: argparse.Namespace):
    """按参数创建模型客户端。"""
    if args.provider is None and args.model is None:
        return None
    return create_chat_client(provider=args.provider, model=args.model)


def main(argv: list[str] | None = None) -> None:
    """命令行主入口。"""
    args = parse_args(argv)
    auto_config = build_auto_config_from_args(args)
    research_config = build_research_config_from_args(args)
    llm_client = build_llm_client_from_args(args)

    print("A 股 TradingAgents 自动模拟交易入口")
    print("当前模式：长期循环" if args.loop else "当前模式：只跑一轮")
    print(f"每轮最多分析：{args.max_candidates} 只")
    print("自动买入范围：仅沪深主板普通 A 股")
    print(f"模拟盘：{'关闭' if args.no_paper_trading else '开启'}")
    if research_config.enable_paper_trading:
        print(f"模拟账户：{research_config.paper_trading_ledger_path}")
    if auto_config.execute_only_during_trading_hours:
        print("成交限制：只在 A 股交易时段写入模拟成交")
    else:
        print("成交限制：允许非交易时段按最近价格模拟成交")
    print()

    trader = AutoPaperTrader(
        auto_config=auto_config,
        research_config=research_config,
        llm_client=llm_client,
        temperature=args.temperature,
        max_debate_rounds=args.debate_rounds,
        max_risk_discuss_rounds=args.risk_rounds,
    )
    try:
        trader.run_forever()
    except Exception as error:
        print()
        print(render_runtime_error(error))
        raise SystemExit(1) from error


def render_runtime_error(error: Exception) -> str:
    """把自动交易运行错误转换成命令行友好的中文提示。"""
    if isinstance(error, LLMAPIError):
        return "\n".join(
            [
                "========== 模型调用失败 ==========",
                str(error),
                "",
                "处理办法：",
                "1. 如果是 HTTP 402，请检查模型服务商账户余额、计费状态和额度。",
                "2. 如果你有其他模型 API，可以用 --provider openai / kimi / gemini 切换。",
                "3. 如果只想先测试候选发现，可以加 --no-paper-trading 或减少 --max-candidates。",
            ]
        )

    return "\n".join(
        [
            "========== 自动交易运行失败 ==========",
            str(error),
            "",
            "如果这是模型接口错误，请先检查 API Key、余额、模型名和网络。",
        ]
    )


if __name__ == "__main__":
    main()
