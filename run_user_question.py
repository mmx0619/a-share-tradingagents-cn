"""A 股 TradingAgents 自然语言运行入口。

这个文件是给人直接运行的入口。

你可以在终端里运行：

    python run_user_question.py

然后输入：

    大唐发电行情怎么样
    帮我看看京东方A能不能买

程序会做这些事：

    1. 识别用户问题；
    2. 识别股票名称和股票代码；
    3. 调用完整 TradingAgents 分析链路；
    4. 保存最终 Markdown 报告；
    5. 在终端输出简短结论和报告路径。

注意：
    这个入口默认使用系统环境变量里的 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from tradingagents_cn.graph import (
    CheckpointThreadInfo,
    ResearchInputConfig,
    list_checkpoint_thread_ids,
    list_checkpoint_threads,
    run_user_question_pipeline,
)
from tradingagents_cn.dataflows.vendor_config import (
    default_data_vendors,
    default_tool_vendors,
    normalize_selected_analysts,
    normalize_vendor_overrides,
)
from tradingagents_cn.llm.errors import LLMAPIError
from tradingagents_cn.llm.factory import create_chat_client


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    这个入口是给人日常使用的，所以参数分成几类：
        - 问题输入；
        - 模型选择；
        - 分析深度；
        - checkpoint/resume；
        - 辅助查看命令。
    """
    parser = argparse.ArgumentParser(description="A 股 TradingAgents 自然语言入口")

    parser.add_argument("-q", "--question", default=None, help="直接传入用户问题，不再交互式 input")
    parser.add_argument("--date", default=None, help="分析日期，格式 YYYY-MM-DD；不传默认今天")

    parser.add_argument(
        "--provider",
        default=None,
        choices=["deepseek", "openai", "kimi", "gemini"],
        help="选择大模型服务商；不传则读取环境变量或默认 deepseek",
    )
    parser.add_argument("--model", default=None, help="指定模型名；不传则使用 provider 默认模型")
    parser.add_argument("--temperature", type=float, default=0.2, help="模型随机性，默认 0.2")

    parser.add_argument("--debate-rounds", type=int, default=1, help="多空辩论轮数，默认 1")
    parser.add_argument("--risk-rounds", type=int, default=1, help="风险辩论轮数，默认 1")
    parser.add_argument("--history-days", type=int, default=420, help="历史行情自然日长度，默认 420")
    parser.add_argument("--news-max-items", type=int, default=2, help="新闻最多条数，默认 2")
    parser.add_argument("--fundamentals-max-rows", type=int, default=2, help="财务表最多行数，默认 2")
    parser.add_argument("--fundamentals-max-columns", type=int, default=20, help="财务表最多列数，默认 20")
    parser.add_argument("--benchmark-symbol", default="000300", help="记忆复盘基准指数代码，默认 000300")
    parser.add_argument("--benchmark-name", default="沪深300", help="记忆复盘基准名称，默认 沪深300")
    parser.add_argument("--memory-holding-days", type=int, default=5, help="pending 决策持有多少期后复盘，默认 5 个交易日")
    parser.add_argument("--resolve-all-pending-memory", action="store_true", help="运行时尝试复盘记忆日志里所有 pending 记录")
    parser.add_argument("--paper-trading", action="store_true", help="启用模拟盘自动交易，只写本地账户，不真实下单")
    parser.add_argument("--paper-ledger", default="outputs/paper_trading/account.json", help="模拟盘账户 JSON 路径")
    parser.add_argument("--paper-initial-cash", type=float, default=10000.0, help="模拟盘初始资金，默认 10000")
    parser.add_argument("--paper-max-position-pct", type=float, default=0.20, help="单只股票模拟仓位上限，默认 0.20")
    parser.add_argument("--paper-min-trade-amount", type=float, default=1000.0, help="模拟盘最小成交金额，默认 1000")
    parser.add_argument("--paper-review-days", type=int, default=5, help="模拟成交持有多少期后复盘，默认 5 个交易日")
    parser.add_argument("--no-paper-review", action="store_true", help="启用模拟盘时，不自动复盘历史 pending 模拟成交")
    parser.add_argument(
        "--analysts",
        default="market,sentiment,news,fundamentals",
        help="启用哪些 Analyst，逗号分隔，例如 market,sentiment,news,fundamentals",
    )
    parser.add_argument("--sentiment-max-items", type=int, default=12, help="情绪材料最多条数，默认 12")
    parser.add_argument(
        "--sentiment-sources",
        default="eastmoney,xueqiu,tonghuashun,taoguba",
        help="情绪源，逗号分隔：eastmoney,xueqiu,tonghuashun,taoguba",
    )

    parser.add_argument("--no-realtime", action="store_true", help="关闭实时/近实时行情获取")
    parser.add_argument("--no-news", action="store_true", help="关闭新闻获取")
    parser.add_argument("--no-sentiment", action="store_true", help="关闭独立 Sentiment Analyst")
    parser.add_argument("--no-fundamentals", action="store_true", help="关闭基本面获取")
    parser.add_argument("--no-deep-screening", action="store_true", help="选股问题只做初筛，不跑深度单股链路")
    parser.add_argument("--deep-top-n", type=int, default=3, help="深度选股最多分析几只候选股，默认 3")
    parser.add_argument("--rule-router", action="store_true", help="不用大模型路由，改用规则路由")

    parser.add_argument("--output-dir", default="outputs/user_questions", help="最终报告输出目录")
    parser.add_argument("--no-full-state", action="store_true", help="不保存 full_state.json 完整运行状态")
    parser.add_argument("--full-state-dir", default="outputs/run_states", help="full_state.json 输出目录")
    parser.add_argument(
        "--data-vendor",
        action="append",
        default=[],
        help="覆盖数据源配置，格式 key=value，例如 market_data=akshare",
    )
    parser.add_argument(
        "--tool-vendor",
        action="append",
        default=[],
        help="覆盖工具源配置，格式 key=value，例如 sentiment_sources=eastmoney,xueqiu",
    )
    parser.add_argument("--resume", action="store_true", help="使用同一个 thread_id 从 SQLite checkpoint 继续")
    parser.add_argument("--thread-id", default=None, help="指定 LangGraph checkpoint thread_id")
    parser.add_argument("--checkpoint-db", default=None, help="指定 SQLite checkpoint 数据库路径")

    parser.add_argument("--list-checkpoints", action="store_true", help="列出 SQLite checkpoint 中已有 thread_id")
    parser.add_argument("--list-reports", action="store_true", help="列出报告目录里的 final_report.md 文件")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    return build_arg_parser().parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ResearchInputConfig:
    """把命令行参数转换成研究输入配置。"""
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
        enable_paper_trading=args.paper_trading,
        paper_trading_ledger_path=args.paper_ledger,
        paper_trading_initial_cash=args.paper_initial_cash,
        paper_trading_max_single_position_pct=args.paper_max_position_pct,
        paper_trading_min_trade_amount=args.paper_min_trade_amount,
        paper_trading_review_pending=not args.no_paper_review,
        paper_trading_review_holding_days=args.paper_review_days,
    )


def build_llm_client_from_args(args: argparse.Namespace):
    """根据命令行参数创建大模型客户端。

    如果用户没有指定 provider/model，就返回 None。
    这样下游仍然使用默认 create_chat_client() 逻辑。
    """
    if args.provider is None and args.model is None:
        return None
    return create_chat_client(provider=args.provider, model=args.model)


def list_report_paths(output_dir: str | Path) -> list[Path]:
    """列出报告目录中的最终报告文件。"""
    directory = Path(output_dir)
    if not directory.exists():
        return []
    reports = list(directory.glob("*_final_report.md"))
    return sorted(reports, key=lambda path: path.stat().st_mtime, reverse=True)


def render_checkpoint_thread_list(thread_ids: list[str]) -> str:
    """渲染 checkpoint thread 列表。"""
    if not thread_ids:
        return "暂无 SQLite checkpoint thread。"
    lines = ["SQLite checkpoint thread 列表："]
    lines.extend(f"- {thread_id}" for thread_id in thread_ids)
    return "\n".join(lines)


def render_checkpoint_thread_info_list(infos: list[CheckpointThreadInfo]) -> str:
    """渲染 checkpoint thread 详细列表。"""
    if not infos:
        return "暂无 SQLite checkpoint thread。"

    lines = ["SQLite checkpoint thread 列表："]
    for index, info in enumerate(infos, start=1):
        ns_text = info.checkpoint_ns or "默认"
        step_text = "未知" if info.latest_step is None else str(info.latest_step)
        lines.extend(
            [
                f"{index}. thread_id：{info.thread_id}",
                f"   namespace：{ns_text}",
                f"   checkpoint 数量：{info.checkpoint_count}",
                f"   最新 step：{step_text}",
                f"   最新 source：{info.latest_source}",
                f"   最新 checkpoint_id：{info.latest_checkpoint_id}",
                f"   续跑命令：--resume --thread-id {info.thread_id}",
            ]
        )
    return "\n".join(lines)


def render_report_list(report_paths: list[Path]) -> str:
    """渲染最终报告文件列表。"""
    if not report_paths:
        return "暂无最终报告文件。"
    lines = ["最终报告列表："]
    lines.extend(f"- {path}" for path in report_paths)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """命令行主入口。

    这个函数就是你调试时最应该打断点的地方。

    执行顺序：
        main()
          -> input(...)
          -> run_user_question_pipeline(...)
          -> print(...)
    """
    args = parse_args(argv)

    if args.list_checkpoints:
        print(render_checkpoint_thread_info_list(list_checkpoint_threads(args.checkpoint_db)))
        return

    if args.list_reports:
        print(render_report_list(list_report_paths(args.output_dir)))
        return

    print("A 股 TradingAgents 自然语言入口")
    print("示例：大唐发电行情怎么样 / 帮我看看京东方A能不能买")
    if args.resume:
        print("当前模式：从 SQLite checkpoint 继续运行")
    if args.provider:
        print(f"模型服务商：{args.provider}")
    if args.model:
        print(f"模型名称：{args.model}")
    if args.paper_trading:
        print(f"模拟盘：已启用，账户文件 {args.paper_ledger}")
    print()

    question = (args.question or input("请输入你的问题：")).strip()
    if not question:
        print("你没有输入问题，程序结束。")
        return

    trade_date = args.date or date.today().strftime("%Y-%m-%d")

    config = build_config_from_args(args)
    llm_client = build_llm_client_from_args(args)

    try:
        result = run_user_question_pipeline(
            question=question,
            trade_date=trade_date,
            config=config,
            llm_client=llm_client,
            temperature=args.temperature,
            max_debate_rounds=args.debate_rounds,
            max_risk_discuss_rounds=args.risk_rounds,
            output_dir=args.output_dir,
            use_llm_router=not args.rule_router,
            enable_deep_stock_screening=not args.no_deep_screening,
            deep_stock_screening_top_n=args.deep_top_n,
            thread_id=args.thread_id,
            resume=args.resume,
            checkpoint_db_path=args.checkpoint_db,
        )
    except Exception as error:
        print()
        print(render_runtime_error(error))
        raise SystemExit(1) from error

    print()
    print("========== 分析完成 ==========")
    if result.thread_id:
        print(f"本次 thread_id：{result.thread_id}")
    print(result.answer)


def render_runtime_error(error: Exception) -> str:
    """把运行时异常转换成命令行友好的中文提示。"""
    if isinstance(error, LLMAPIError):
        return "\n".join(
            [
                "========== 模型调用失败 ==========",
                str(error),
                "",
                "处理办法：",
                "1. 如果是 HTTP 402，请检查模型服务商账户余额、计费状态和额度。",
                "2. 如果你有其他模型 API，可以用 --provider openai / kimi / gemini 切换。",
                "3. 如果只是想先验证路由，可以临时加 --rule-router；完整分析仍需要大模型。",
            ]
        )

    return "\n".join(
        [
            "========== 程序运行失败 ==========",
            str(error),
            "",
            "如果这是模型接口错误，请先检查 API Key、余额、模型名和网络。",
        ]
    )


if __name__ == "__main__":
    main()
