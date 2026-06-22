"""A 股研究工作流输入准备层。

这个文件负责把真实 dataflows 数据组装成 research_workflow 需要的 state。

它做的事情：

    股票代码 + 分析日期
      -> 拉历史日线行情
      -> 计算原版 TradingAgents 技术指标集合
      -> 可选拉实时行情
      -> 可选拉个股新闻
      -> 可选拉基本面材料
      -> 组装成 ResearchWorkflowState

注意：
    这里仍然不调用大模型。
    它只是把真实数据准备好，交给 LangGraph 工作流生成各 Agent Prompt。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from tradingagents_cn.dataflows.data_quality import (
    validate_daily_history,
    validate_realtime_quote,
    validate_stock_news_items,
)
from tradingagents_cn.dataflows.fundamentals import (
    get_balance_sheet,
    get_cashflow,
    get_company_profile,
    get_income_statement,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol
from tradingagents_cn.dataflows.vendor_config import (
    DEFAULT_SELECTED_ANALYSTS,
    default_data_vendors,
    default_tool_vendors,
    resolve_tool_vendor,
)
from tradingagents_cn.dataflows.vendor_router import (
    route_daily_history,
    route_realtime_quote,
    route_stock_news,
)
from tradingagents_cn.graph.research_workflow import ResearchWorkflowState
from tradingagents_cn.indicators import add_tradingagents_indicators


@dataclass
class ResearchInputConfig:
    """研究工作流输入准备配置。

    history_calendar_days:
        往前取多少个自然日的历史行情。
        原版指标里有 close_200_sma，所以默认给 420 个自然日，
        以便尽量覆盖 200 个交易日。

    news_max_items:
        个股新闻最多取多少条。

    fundamentals_max_rows:
        三张财务表最多取多少个报告期行。

    fundamentals_max_columns:
        三张财务表最多保留多少个关键字段列。

    include_realtime / include_news / include_fundamentals:
        控制是否拉对应数据。
        调试时可以关掉较慢的数据源。
    include_sentiment:
        控制是否启用独立 Sentiment Analyst。
    selected_analysts:
        控制主图里实际运行哪些分析员。
        默认是 market、sentiment、news、fundamentals 四个。
    sentiment_sources:
        情绪面公开网页来源，默认同时尝试东方财富股吧、雪球、同花顺股吧、淘股吧。
    data_vendors / tool_vendors:
        数据源和工具源配置。
        当前正式实现以 AKShare 和公开网页为主，保留这个入口是为了后续替换供应商。
    save_full_state / full_state_output_dir:
        是否保存完整运行状态 JSON，以及保存目录。
    benchmark_symbol / benchmark_name:
        记忆复盘时使用的 A 股基准指数，默认沪深300。
    memory_holding_days:
        pending 决策默认持有多少个交易日后做收益复盘。
    resolve_all_pending_memory:
        True 时尝试复盘记忆日志里所有 pending 记录；
        False 时只复盘当前股票。
    enable_paper_trading:
        是否启用模拟盘自动交易。
        默认 False，避免只是生成报告时意外写模拟账户。
    paper_trading_ledger_path:
        模拟盘账户 JSON 文件路径。
    paper_trading_initial_cash:
        模拟盘初始资金。
    paper_trading_max_single_position_pct:
        单只股票在模拟账户里的最大仓位上限。
        真正下单时，还会和程序风控护栏里的 max_position_pct 取更小值。
    paper_trading_min_trade_amount:
        最小模拟成交金额，低于这个金额就不下单。
    paper_trading_review_pending:
        每次启用模拟盘时，是否顺手复盘过去 pending 的模拟成交。
    paper_trading_review_holding_days:
        模拟成交经过多少个交易日后尝试复盘。
    paper_trading_buy_universe:
        模拟盘允许自动买入的股票范围。
        当前固定为 main_board_common，也就是沪深主板普通 A 股。
        创业板、科创板、北交所、ST、退市风险股即使模型说 BUY，也会被跳过。
    """

    history_calendar_days: int = 420
    news_max_items: int = 5
    sentiment_max_items: int = 12
    fundamentals_max_rows: int = 4
    fundamentals_max_columns: int = 30
    include_realtime: bool = True
    include_news: bool = True
    include_sentiment: bool = True
    include_fundamentals: bool = True
    selected_analysts: tuple[str, ...] = DEFAULT_SELECTED_ANALYSTS
    sentiment_sources: str = "eastmoney,xueqiu,tonghuashun,taoguba"
    data_vendors: dict[str, str] = field(default_factory=default_data_vendors)
    tool_vendors: dict[str, str] = field(default_factory=default_tool_vendors)
    save_full_state: bool = True
    full_state_output_dir: str = "outputs/run_states"
    benchmark_symbol: str = "000300"
    benchmark_name: str = "沪深300"
    memory_holding_days: int = 5
    resolve_all_pending_memory: bool = False
    enable_paper_trading: bool = False
    paper_trading_ledger_path: str = "outputs/paper_trading/account.json"
    paper_trading_initial_cash: float = 10000.0
    paper_trading_max_single_position_pct: float = 0.20
    paper_trading_min_trade_amount: float = 1000.0
    paper_trading_review_pending: bool = True
    paper_trading_review_holding_days: int = 5
    paper_trading_buy_universe: str = "main_board_common"


def build_research_initial_state(
    symbol: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
) -> ResearchWorkflowState:
    """构造正式研究工作流的初始 state。

    参数：
        symbol:
            A 股股票代码，例如 002361。

        trade_date:
            分析日期，格式 YYYY-MM-DD。
            如果不传，使用今天日期。

        config:
            输入准备配置。

    返回：
        ResearchWorkflowState。

    失败策略：
        历史行情和技术指标是 Market Agent 必需数据，失败就抛错。
        实时行情、新闻、基本面是外围材料，失败会写入 data_errors，
        工作流仍然可以继续生成其它 Agent Prompt。
    """
    input_config = config or ResearchInputConfig()
    normalized_symbol = normalize_cn_symbol(symbol)
    actual_trade_date = trade_date or date.today().strftime("%Y-%m-%d")
    start_date = calculate_history_start_date(
        actual_trade_date,
        input_config.history_calendar_days,
    )

    data_errors: list[str] = []

    daily_history = route_daily_history(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=actual_trade_date,
        vendor=resolve_tool_vendor(input_config, "daily_history"),
    )
    data_errors.extend(
        validate_daily_history(
            daily_history,
            trade_date=actual_trade_date,
        )
    )
    indicator_data = add_tradingagents_indicators(daily_history)

    realtime_quote = None
    if input_config.include_realtime:
        try:
            realtime_quote = route_realtime_quote(
                normalized_symbol,
                vendor=resolve_tool_vendor(input_config, "realtime_quote"),
            )
            data_errors.extend(validate_realtime_quote(realtime_quote))
        except Exception as error:
            data_errors.append(f"实时行情获取失败：{error}")

    stock_news_items = []
    if input_config.include_news:
        try:
            stock_news_items = route_stock_news(
                normalized_symbol,
                max_items=input_config.news_max_items,
                vendor=resolve_tool_vendor(input_config, "stock_news"),
            )
            data_errors.extend(
                validate_stock_news_items(
                    stock_news_items,
                    trade_date=actual_trade_date,
                )
            )
        except Exception as error:
            data_errors.append(f"个股新闻获取失败：{error}")

    company_profile_text = None
    balance_sheet_text = None
    cashflow_text = None
    income_statement_text = None

    if input_config.include_fundamentals:
        try:
            company_profile_text = get_company_profile(normalized_symbol)
        except Exception as error:
            data_errors.append(f"公司基本资料获取失败：{error}")

        try:
            balance_sheet_text = get_balance_sheet(
                normalized_symbol,
                max_rows=input_config.fundamentals_max_rows,
                max_columns=input_config.fundamentals_max_columns,
            )
        except Exception as error:
            data_errors.append(f"资产负债表获取失败：{error}")

        try:
            cashflow_text = get_cashflow(
                normalized_symbol,
                max_rows=input_config.fundamentals_max_rows,
                max_columns=input_config.fundamentals_max_columns,
            )
        except Exception as error:
            data_errors.append(f"现金流量表获取失败：{error}")

        try:
            income_statement_text = get_income_statement(
                normalized_symbol,
                max_rows=input_config.fundamentals_max_rows,
                max_columns=input_config.fundamentals_max_columns,
            )
        except Exception as error:
            data_errors.append(f"利润表获取失败：{error}")

    return ResearchWorkflowState(
        symbol=normalized_symbol,
        trade_date=actual_trade_date,
        indicator_data=indicator_data,
        realtime_quote=realtime_quote,
        stock_news_items=stock_news_items,
        macro_news_text=None,
        company_profile_text=company_profile_text,
        balance_sheet_text=balance_sheet_text,
        cashflow_text=cashflow_text,
        income_statement_text=income_statement_text,
        data_errors=data_errors,
    )


def calculate_history_start_date(trade_date: str, history_calendar_days: int) -> str:
    """根据分析日期计算历史行情开始日期。

    为什么用自然日而不是交易日？
        AKShare 历史行情接口按日期区间取数。
        这里先用自然日往前推，接口会自动只返回交易日。
    """
    end = datetime.strptime(trade_date, "%Y-%m-%d").date()
    start = end - timedelta(days=history_calendar_days)
    return start.strftime("%Y-%m-%d")
