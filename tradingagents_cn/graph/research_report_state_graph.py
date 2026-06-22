"""完整 A 股 TradingAgents StateGraph。

早期版本的完整分析链路主要写在普通 Python 函数里：

    run_research_report_pipeline(...)

这个文件把同样的业务流程拆成 LangGraph StateGraph 节点。

图结构：

    prepare_prompts
      -> market_agent
      -> sentiment_agent
      -> news_agent
      -> fundamentals_agent
      -> summary_agent
      -> bull_researcher
      -> bear_researcher
      -> ...按 max_debate_rounds 条件路由
      -> research_manager
      -> trader
      -> aggressive_risk_analyst
      -> conservative_risk_analyst
      -> neutral_risk_analyst
      -> ...按 max_risk_discuss_rounds 条件路由
      -> portfolio_manager
      -> END

这样做的意义：

    1. 每一步都有明确节点名；
    2. 后续可以对每个节点做 checkpoint/resume；
    3. 更接近原版 TradingAgents 的工程结构；
    4. 调试时可以清楚知道程序运行到哪个节点。
"""

from __future__ import annotations

from datetime import date
from typing import Any, TypedDict

from tradingagents_cn.agents import (
    FUNDAMENTALS_ANALYST_SYSTEM_PROMPT,
    MARKET_ANALYST_SYSTEM_PROMPT,
    MARKET_INDICATOR_CATALOG,
    NEWS_ANALYST_SYSTEM_PROMPT,
    SENTIMENT_ANALYST_SYSTEM_PROMPT,
    ResearchPlan,
    TraderProposal,
    PortfolioDecision,
    RiskAssessment,
    DebateArgument,
    build_aggressive_risk_context,
    build_bear_researcher_context,
    build_bull_researcher_context,
    build_conservative_risk_context,
    build_fallback_debate_argument,
    build_fallback_portfolio_decision,
    build_fallback_research_plan,
    build_fallback_risk_assessment,
    build_fallback_trader_proposal,
    build_neutral_risk_context,
    build_portfolio_manager_context,
    build_research_manager_context,
    build_summary_agent_context,
    build_trader_agent_context,
    parse_portfolio_decision_from_text,
    parse_research_plan_from_text,
    parse_trader_proposal_from_text,
    render_debate_argument,
    render_portfolio_decision,
    render_research_plan,
    render_risk_assessment,
    render_trader_proposal,
)
from tradingagents_cn.graph.checkpointing import (
    create_memory_checkpointer,
    create_sqlite_checkpointer,
    has_checkpoint_for_thread,
)
from tradingagents_cn.graph.analyst_tool_calling import (
    convert_messages_for_debug,
    run_analyst_tool_calling_report,
)
from tradingagents_cn.graph.propagation import ResearchGraphPropagator
from tradingagents_cn.graph.progress import emit_progress
from tradingagents_cn.graph.reflection import Reflector
from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.dataflows.vendor_config import (
    filter_sentiment_analyst,
    normalize_selected_analysts,
    resolve_data_vendor,
    resolve_tool_vendor,
)
from tradingagents_cn.graph.research_pipeline import ResearchPromptPipelineResult
from tradingagents_cn.graph.research_report_pipeline import (
    ChatModelClient,
    ResearchReportPipelineResult,
    DebateRunResult,
    RiskDebateRunResult,
    append_debate_argument,
    build_agent_messages,
    build_memory_decision_text,
    call_agent_report,
)
from tradingagents_cn.graph.signal_processing import (
    build_risk_guardrail_decision,
    process_portfolio_rating,
)
from tradingagents_cn.graph.run_state_logging import save_full_state_json
from tradingagents_cn.graph.setup import AnalystNodeSpec, GraphSetup
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.llm.structured_output import call_structured_output
from tradingagents_cn.memory import TradingMemoryLog
from tradingagents_cn.paper_trading import (
    PaperTradingConfig,
    run_paper_trading_from_result,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol
from tradingagents_cn.tools.akshare_tools import akshare_realtime_quote, get_market_technical_snapshot
from tradingagents_cn.tools.news_tools import akshare_stock_news
from tradingagents_cn.tools.sentiment_tools import get_stock_sentiment
from tradingagents_cn.tools.announcement_tools import get_stock_announcements_tool
from tradingagents_cn.tools.fundamentals_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)


class ResearchReportGraphState(TypedDict, total=False):
    """完整研究报告图状态。

    这个 state 会在各个节点之间不断追加字段。
    你在调试器里看它，就能知道程序目前跑到了哪一步。
    """

    symbol: str
    trade_date: str | None
    config: ResearchInputConfig | None
    llm_client: ChatModelClient | None
    temperature: float
    max_debate_rounds: int
    max_risk_discuss_rounds: int
    enable_memory: bool
    memory_log_path: str | None
    selected_analysts: tuple[str, ...]

    prompt_result: ResearchPromptPipelineResult
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    summary_report: str
    debate_result: Any
    debate_history: str
    debate_count: int
    last_debate_speaker: str
    last_bull_argument: str
    last_bear_argument: str
    research_plan: Any
    investment_plan: str
    trader_proposal: Any
    trader_plan: str
    risk_result: Any
    risk_history: str
    risk_count: int
    latest_risk_speaker: str
    last_aggressive_argument: str
    last_conservative_argument: str
    last_neutral_argument: str
    risk_assessments: list[Any]
    portfolio_decision: Any
    final_trade_decision: str
    trade_signal: Any
    risk_guardrail: Any
    messages_by_agent: dict[str, list[dict[str, Any]]]
    tool_call_trace: list[dict[str, Any]]
    tool_call_stats: dict[str, Any]
    reflection_summary: dict[str, Any]
    paper_trading_result: dict[str, Any]


def get_state_client(state: ResearchReportGraphState):
    """从 state 里取模型客户端，没有就用 LLM factory 创建。"""
    return state.get("llm_client") or create_chat_client()


def get_configured_analysts(config: ResearchInputConfig) -> tuple[str, ...]:
    """读取并规范化本次要运行的 Analyst 列表。"""
    selected = normalize_selected_analysts(config.selected_analysts)
    return filter_sentiment_analyst(selected, include_sentiment=config.include_sentiment)


def missing_report(label: str) -> str:
    """给未启用的 Analyst 生成明确占位文本。"""
    return f"本次未启用 {label}，没有生成对应报告。"


def get_report_or_missing(
    state: ResearchReportGraphState,
    key: str,
    label: str,
) -> str:
    """读取某份报告；如果节点未运行，返回可读的缺失说明。"""
    report = state.get(key)
    if isinstance(report, str) and report.strip():
        return report
    return missing_report(label)


def build_debate_result_from_state(state: ResearchReportGraphState) -> DebateRunResult:
    """从 StateGraph 状态里组装多空辩论结果。"""
    return DebateRunResult(
        last_bull_argument=state.get("last_bull_argument", ""),
        last_bear_argument=state.get("last_bear_argument", ""),
        debate_history=state.get("debate_history", ""),
        messages_by_agent={
            key: value
            for key, value in state.get("messages_by_agent", {}).items()
            if key.startswith("bull_round_") or key.startswith("bear_round_")
        },
    )


def build_risk_result_from_state(state: ResearchReportGraphState) -> RiskDebateRunResult:
    """从 StateGraph 状态里组装风险辩论结果。"""
    return RiskDebateRunResult(
        last_aggressive_argument=state.get("last_aggressive_argument", ""),
        last_conservative_argument=state.get("last_conservative_argument", ""),
        last_neutral_argument=state.get("last_neutral_argument", ""),
        risk_history=state.get("risk_history", ""),
        risk_assessments=list(state.get("risk_assessments", [])),
        messages_by_agent={
            key: value
            for key, value in state.get("messages_by_agent", {}).items()
            if key.startswith("aggressive_risk_round_")
            or key.startswith("conservative_risk_round_")
            or key.startswith("neutral_risk_round_")
        },
    )


def prepare_prompts_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """准备最小图状态。

    Tool Calling 版不在这里提前获取行情、新闻、基本面。
    这里只标准化股票代码和日期，真正取数交给 Analyst 自己调用工具。
    """
    emit_progress("正在准备分析状态、交易日期和本次启用的分析员。")
    config = state.get("config") or ResearchInputConfig()
    normalized_symbol = normalize_cn_symbol(state["symbol"])
    actual_trade_date = state.get("trade_date") or date.today().strftime("%Y-%m-%d")
    selected_analysts = tuple(
        state.get("selected_analysts") or get_configured_analysts(config)
    )
    prompt_result = ResearchPromptPipelineResult(
        final_state={
            "symbol": normalized_symbol,
            "trade_date": actual_trade_date,
            "data_errors": [],
            "selected_analysts": selected_analysts,
            "data_vendors": dict(config.data_vendors),
            "tool_vendors": dict(config.tool_vendors),
        }
    )
    return {
        **state,
        "symbol": normalized_symbol,
        "trade_date": actual_trade_date,
        "config": config,
        "selected_analysts": selected_analysts,
        "prompt_result": prompt_result,
        "messages_by_agent": {},
        "tool_call_trace": [],
        "tool_call_stats": build_empty_tool_call_stats(),
        "reflection_summary": {},
        "paper_trading_result": {},
        "debate_history": "",
        "debate_count": 0,
        "last_debate_speaker": "",
        "last_bull_argument": "",
        "last_bear_argument": "",
        "risk_history": "",
        "risk_count": 0,
        "latest_risk_speaker": "",
        "last_aggressive_argument": "",
        "last_conservative_argument": "",
        "last_neutral_argument": "",
        "risk_assessments": [],
    }


def market_agent_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Market Agent 节点。

    这个节点已经改成 Tool Calling：
        Market Agent 先调用技术快照/实时行情工具，
        再基于工具结果写技术面报告。
    """
    emit_progress("正在运行技术面分析：获取行情、技术指标和实时价格。")
    final_state = state["prompt_result"].final_state
    config = state.get("config") or ResearchInputConfig()
    system_prompt = MARKET_ANALYST_SYSTEM_PROMPT.format(
        indicator_catalog=MARKET_INDICATOR_CATALOG
    )
    data_vendor = resolve_data_vendor(config, "market_data")
    tool_vendor = resolve_tool_vendor(config, "market_agent")
    daily_history_vendor = resolve_tool_vendor(config, "daily_history", tool_vendor)
    realtime_quote_vendor = resolve_tool_vendor(config, "realtime_quote", tool_vendor)
    task_prompt = f"""当前分析股票：{final_state["symbol"]}
当前分析日期：{final_state["trade_date"]}
行情/技术数据源配置：{data_vendor}
Market Agent 工具源配置：{tool_vendor}
历史行情 vendor：{daily_history_vendor}
实时行情 vendor：{realtime_quote_vendor}

请先调用工具获取市场技术面数据，再撰写 A 股市场技术面分析报告。

工具调用要求：
1. 必须调用 get_market_technical_snapshot，获取已校验历史行情和技术指标快照。
   调用时 vendor 使用：{daily_history_vendor}。
2. 如需近实时价格，请调用 akshare_realtime_quote。
   调用时 vendor 使用：{realtime_quote_vendor}。
3. 不要编造工具没有返回的价格、成交量、指标数值。
4. 不要直接给最终买入/卖出结论。

建议历史区间自然日：{config.history_calendar_days}
"""
    result = run_analyst_tool_calling_report(
        agent_name="Market Agent",
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        tools=[get_market_technical_snapshot, akshare_realtime_quote],
        llm_client=get_state_client(state),
        force_first_tool_name="get_market_technical_snapshot",
        thread_id=f"market-agent-{final_state['symbol']}-{final_state['trade_date']}",
    )
    return append_agent_result(
        state,
        "market",
        convert_messages_for_debug(result.messages),
        {"market_report": result.report},
        tool_result=result,
    )


def sentiment_agent_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Sentiment Agent 节点，使用 Tool Calling 获取 A 股社区情绪材料。"""
    emit_progress("正在运行情绪面分析：获取股吧、社区和舆情材料。")
    final_state = state["prompt_result"].final_state
    config = state.get("config") or ResearchInputConfig()
    data_vendor = resolve_data_vendor(config, "sentiment")
    tool_vendor = resolve_tool_vendor(config, "sentiment_agent")
    sources = resolve_tool_vendor(config, "sentiment_sources", config.sentiment_sources)
    task_prompt = f"""当前分析股票：{final_state["symbol"]}
当前分析日期：{final_state["trade_date"]}
情绪数据源配置：{data_vendor}
Sentiment Agent 工具源配置：{tool_vendor}

请先调用工具获取 A 股社区情绪材料，再撰写 A 股情绪面分析报告。

工具调用要求：
1. 必须调用 get_stock_sentiment。
2. 调用 get_stock_sentiment 时，symbol 使用 {final_state["symbol"]}。
3. max_items 建议使用 {config.sentiment_max_items}。
4. vendor 使用：{tool_vendor}。
5. sources 建议使用：{sources}。
6. 如果某个情绪源没有返回内容，要明确说明这是数据缺口，不要编造社区观点。
7. 不要把“情绪看多/看空”直接等同于股价一定上涨/下跌。
8. 不要直接给最终买入/卖出结论。
"""
    result = run_analyst_tool_calling_report(
        agent_name="Sentiment Agent",
        system_prompt=SENTIMENT_ANALYST_SYSTEM_PROMPT,
        task_prompt=task_prompt,
        tools=[get_stock_sentiment],
        llm_client=get_state_client(state),
        force_first_tool_name="get_stock_sentiment",
        thread_id=f"sentiment-agent-{final_state['symbol']}-{final_state['trade_date']}",
    )
    return append_agent_result(
        state,
        "sentiment",
        convert_messages_for_debug(result.messages),
        {"sentiment_report": result.report},
        tool_result=result,
    )


def news_agent_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """News Agent 节点，使用 Tool Calling 获取新闻和公告材料。"""
    emit_progress("正在运行新闻面分析：获取个股新闻和公告材料。")
    final_state = state["prompt_result"].final_state
    config = state.get("config") or ResearchInputConfig()
    data_vendor = resolve_data_vendor(config, "news")
    tool_vendor = resolve_tool_vendor(config, "news_agent")
    stock_news_vendor = resolve_tool_vendor(config, "stock_news", tool_vendor)
    announcement_vendor = resolve_tool_vendor(config, "announcement_tool")
    task_prompt = f"""当前分析股票：{final_state["symbol"]}
当前分析日期：{final_state["trade_date"]}
新闻数据源配置：{data_vendor}
News Agent 工具源配置：{tool_vendor}
个股新闻 vendor：{stock_news_vendor}
公告 vendor：{announcement_vendor}

请先调用工具获取新闻和公告材料，再撰写 A 股新闻面分析报告。

工具调用要求：
1. 必须调用 akshare_stock_news 获取个股新闻。
   调用时 vendor 使用：{stock_news_vendor}，trade_date 使用：{final_state["trade_date"]}。
2. 如需核实上市公司正式披露，请调用 get_stock_announcements_tool。
   调用时 vendor 使用：{announcement_vendor}。
3. 新闻较旧、公告缺失或缺少实质内容时，必须明确说明。
4. 不要直接给最终买入/卖出结论。

建议新闻条数：{config.news_max_items}
"""
    result = run_analyst_tool_calling_report(
        agent_name="News Agent",
        system_prompt=NEWS_ANALYST_SYSTEM_PROMPT,
        task_prompt=task_prompt,
        tools=[akshare_stock_news, get_stock_announcements_tool],
        llm_client=get_state_client(state),
        force_first_tool_name="akshare_stock_news",
        thread_id=f"news-agent-{final_state['symbol']}-{final_state['trade_date']}",
    )
    return append_agent_result(
        state,
        "news",
        convert_messages_for_debug(result.messages),
        {"news_report": result.report},
        tool_result=result,
    )


def fundamentals_agent_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Fundamentals Agent 节点，使用 Tool Calling 获取基本面材料。"""
    emit_progress("正在运行基本面分析：获取公司资料、财务摘要和三张表。")
    final_state = state["prompt_result"].final_state
    config = state.get("config") or ResearchInputConfig()
    data_vendor = resolve_data_vendor(config, "fundamentals")
    tool_vendor = resolve_tool_vendor(config, "fundamentals_agent")
    fundamentals_vendor = resolve_tool_vendor(config, "fundamentals", tool_vendor)
    balance_sheet_vendor = resolve_tool_vendor(config, "balance_sheet", tool_vendor)
    cashflow_vendor = resolve_tool_vendor(config, "cashflow", tool_vendor)
    income_statement_vendor = resolve_tool_vendor(config, "income_statement", tool_vendor)
    task_prompt = f"""当前分析股票：{final_state["symbol"]}
当前分析日期：{final_state["trade_date"]}
基本面数据源配置：{data_vendor}
Fundamentals Agent 工具源配置：{tool_vendor}
综合基本面 vendor：{fundamentals_vendor}
资产负债表 vendor：{balance_sheet_vendor}
现金流量表 vendor：{cashflow_vendor}
利润表 vendor：{income_statement_vendor}

请先调用工具获取公司基本面和财务材料，再撰写 A 股基本面分析报告。

工具调用要求：
1. 必须调用 get_fundamentals 获取公司资料和财务历史摘要。
   调用时 vendor 使用：{fundamentals_vendor}。
2. 如需进一步核实三张表，请调用 get_balance_sheet、get_cashflow、get_income_statement。
   三张表对应 vendor 分别使用：{balance_sheet_vendor}、{cashflow_vendor}、{income_statement_vendor}。
3. 如果材料缺失，必须明确说明缺失项。
4. 不要编造工具没有返回的财务数据。
5. 不要直接给最终买入/卖出结论。

建议财务表行数：{config.fundamentals_max_rows}
建议财务表列数：{config.fundamentals_max_columns}
"""
    result = run_analyst_tool_calling_report(
        agent_name="Fundamentals Agent",
        system_prompt=FUNDAMENTALS_ANALYST_SYSTEM_PROMPT,
        task_prompt=task_prompt,
        tools=[get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement],
        llm_client=get_state_client(state),
        force_first_tool_name="get_fundamentals",
        thread_id=f"fundamentals-agent-{final_state['symbol']}-{final_state['trade_date']}",
    )
    return append_agent_result(
        state,
        "fundamentals",
        convert_messages_for_debug(result.messages),
        {"fundamentals_report": result.report},
        tool_result=result,
    )


def summary_agent_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Summary Agent 节点。"""
    emit_progress("正在汇总技术面、情绪面、新闻面和基本面材料。")
    final_state = state["prompt_result"].final_state
    context = build_summary_agent_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        sentiment_report=get_report_or_missing(state, "sentiment_report", "Sentiment Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
    )
    messages = build_agent_messages("Summary Agent", context.prompt)
    report = call_agent_report(
        llm_client=get_state_client(state),
        messages=messages,
        temperature=state.get("temperature", 0.2),
    )
    return append_agent_result(state, "summary", messages, {"summary_report": report})


def bull_researcher_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Bull Researcher 节点：多头研究员单次发言。"""
    final_state = state["prompt_result"].final_state
    count = int(state.get("debate_count", 0))
    round_index = count // 2 + 1
    emit_progress(f"正在进行多空辩论：多头研究员第 {round_index} 轮发言。")
    context = build_bull_researcher_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        summary_report=state["summary_report"],
        debate_history=state.get("debate_history", ""),
        last_bear_argument=state.get("last_bear_argument", ""),
    )
    messages = build_agent_messages(
        agent_name=f"Bull Researcher 第 {round_index} 轮",
        prompt=context.prompt,
    )
    structured_result = call_structured_output(
        llm_client=get_state_client(state),
        messages=messages,
        schema_model=DebateArgument,
        fallback_factory=lambda error: build_fallback_debate_argument("bull", error),
        temperature=0.0,
        max_retries=2,
    )
    argument_body = render_debate_argument(structured_result.value)
    last_bull_argument = f"Bull Researcher Round {round_index}: {argument_body}"
    debate_history = append_debate_argument(
        state.get("debate_history", ""),
        last_bull_argument,
    )
    next_state = append_agent_result(
        state,
        f"bull_round_{round_index}",
        structured_result.messages,
        {
            "debate_history": debate_history,
            "debate_count": count + 1,
            "last_debate_speaker": "bull",
            "last_bull_argument": last_bull_argument,
        },
    )
    return {
        **next_state,
        "debate_result": build_debate_result_from_state(next_state),
    }


def bear_researcher_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Bear Researcher 节点：空头研究员单次发言。"""
    final_state = state["prompt_result"].final_state
    count = int(state.get("debate_count", 0))
    round_index = count // 2 + 1
    emit_progress(f"正在进行多空辩论：空头研究员第 {round_index} 轮发言。")
    context = build_bear_researcher_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        summary_report=state["summary_report"],
        debate_history=state.get("debate_history", ""),
        last_bull_argument=state.get("last_bull_argument", ""),
    )
    messages = build_agent_messages(
        agent_name=f"Bear Researcher 第 {round_index} 轮",
        prompt=context.prompt,
    )
    structured_result = call_structured_output(
        llm_client=get_state_client(state),
        messages=messages,
        schema_model=DebateArgument,
        fallback_factory=lambda error: build_fallback_debate_argument("bear", error),
        temperature=0.0,
        max_retries=2,
    )
    argument_body = render_debate_argument(structured_result.value)
    last_bear_argument = f"Bear Researcher Round {round_index}: {argument_body}"
    debate_history = append_debate_argument(
        state.get("debate_history", ""),
        last_bear_argument,
    )
    next_state = append_agent_result(
        state,
        f"bear_round_{round_index}",
        structured_result.messages,
        {
            "debate_history": debate_history,
            "debate_count": count + 1,
            "last_debate_speaker": "bear",
            "last_bear_argument": last_bear_argument,
        },
    )
    return {
        **next_state,
        "debate_result": build_debate_result_from_state(next_state),
    }


def should_continue_debate(state: ResearchReportGraphState) -> str:
    """根据辩论次数决定继续给谁发言，或进入 Research Manager。"""
    max_rounds = max(1, int(state.get("max_debate_rounds", 1)))
    if int(state.get("debate_count", 0)) >= 2 * max_rounds:
        return "research_manager"
    if state.get("last_debate_speaker") == "bull":
        return "bear_researcher"
    return "bull_researcher"


def research_manager_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Research Manager 节点。"""
    emit_progress("正在由研究经理综合多空辩论，给出研究评级。")
    final_state = state["prompt_result"].final_state
    context = build_research_manager_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        debate_history=state["debate_result"].debate_history,
    )
    messages = build_agent_messages("Research Manager", context.prompt)
    structured_result = call_structured_output(
        llm_client=get_state_client(state),
        messages=messages,
        schema_model=ResearchPlan,
        fallback_factory=build_fallback_research_plan,
        temperature=0.0,
        max_retries=2,
    )
    research_plan = structured_result.value
    investment_plan = render_research_plan(research_plan)
    return append_agent_result(
        state,
        "research_manager",
        structured_result.messages,
        {
            "research_plan": research_plan,
            "investment_plan": investment_plan,
        },
    )


def trader_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Trader Agent 节点。"""
    emit_progress("正在由交易员把研究评级转换为交易预案。")
    final_state = state["prompt_result"].final_state
    context = build_trader_agent_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        investment_plan=state["investment_plan"],
    )
    messages = build_agent_messages("Trader Agent", context.prompt)
    structured_result = call_structured_output(
        llm_client=get_state_client(state),
        messages=messages,
        schema_model=TraderProposal,
        fallback_factory=build_fallback_trader_proposal,
        temperature=0.0,
        max_retries=2,
    )
    trader_proposal = structured_result.value
    trader_plan = render_trader_proposal(trader_proposal)
    return append_agent_result(
        state,
        "trader",
        structured_result.messages,
        {
            "trader_proposal": trader_proposal,
            "trader_plan": trader_plan,
        },
    )


def aggressive_risk_analyst_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Aggressive Risk Analyst 节点：激进风险视角单次发言。"""
    final_state = state["prompt_result"].final_state
    count = int(state.get("risk_count", 0))
    round_index = count // 3 + 1
    emit_progress(f"正在进行风险辩论：激进风险分析员第 {round_index} 轮发言。")
    context = build_aggressive_risk_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        investment_plan=state["investment_plan"],
        trader_plan=state["trader_plan"],
        risk_history=state.get("risk_history", ""),
        last_conservative_argument=state.get("last_conservative_argument", ""),
        last_neutral_argument=state.get("last_neutral_argument", ""),
    )
    return run_single_risk_analyst_node(
        state=state,
        role="aggressive",
        round_index=round_index,
        context_prompt=context.prompt,
        count=count,
    )


def conservative_risk_analyst_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Conservative Risk Analyst 节点：保守风险视角单次发言。"""
    final_state = state["prompt_result"].final_state
    count = int(state.get("risk_count", 0))
    round_index = count // 3 + 1
    emit_progress(f"正在进行风险辩论：保守风险分析员第 {round_index} 轮发言。")
    context = build_conservative_risk_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        investment_plan=state["investment_plan"],
        trader_plan=state["trader_plan"],
        risk_history=state.get("risk_history", ""),
        last_aggressive_argument=state.get("last_aggressive_argument", ""),
        last_neutral_argument=state.get("last_neutral_argument", ""),
    )
    return run_single_risk_analyst_node(
        state=state,
        role="conservative",
        round_index=round_index,
        context_prompt=context.prompt,
        count=count,
    )


def neutral_risk_analyst_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Neutral Risk Analyst 节点：中性风险视角单次发言。"""
    final_state = state["prompt_result"].final_state
    count = int(state.get("risk_count", 0))
    round_index = count // 3 + 1
    emit_progress(f"正在进行风险辩论：中性风险分析员第 {round_index} 轮发言。")
    context = build_neutral_risk_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        investment_plan=state["investment_plan"],
        trader_plan=state["trader_plan"],
        risk_history=state.get("risk_history", ""),
        last_aggressive_argument=state.get("last_aggressive_argument", ""),
        last_conservative_argument=state.get("last_conservative_argument", ""),
    )
    return run_single_risk_analyst_node(
        state=state,
        role="neutral",
        round_index=round_index,
        context_prompt=context.prompt,
        count=count,
    )


def run_single_risk_analyst_node(
    state: ResearchReportGraphState,
    role: str,
    round_index: int,
    context_prompt: str,
    count: int,
) -> ResearchReportGraphState:
    """执行单个风险分析员节点，并把结果写入 state。"""
    display_names = {
        "aggressive": "Aggressive Risk Analyst",
        "conservative": "Conservative Risk Analyst",
        "neutral": "Neutral Risk Analyst",
    }
    message_key_prefix = {
        "aggressive": "aggressive_risk_round",
        "conservative": "conservative_risk_round",
        "neutral": "neutral_risk_round",
    }
    last_argument_keys = {
        "aggressive": "last_aggressive_argument",
        "conservative": "last_conservative_argument",
        "neutral": "last_neutral_argument",
    }
    agent_name = display_names[role]
    messages = build_agent_messages(
        agent_name=f"{agent_name} 第 {round_index} 轮",
        prompt=context_prompt,
    )
    structured_result = call_structured_output(
        llm_client=get_state_client(state),
        messages=messages,
        schema_model=RiskAssessment,
        fallback_factory=lambda error: build_fallback_risk_assessment(role, error),
        temperature=0.0,
        max_retries=2,
    )
    argument_body = render_risk_assessment(structured_result.value)
    last_argument = f"{agent_name} Round {round_index}: {argument_body}"
    risk_history = append_debate_argument(state.get("risk_history", ""), last_argument)
    risk_assessments = list(state.get("risk_assessments", []))
    risk_assessments.append(structured_result.value)
    next_state = append_agent_result(
        state,
        f"{message_key_prefix[role]}_{round_index}",
        structured_result.messages,
        {
            "risk_history": risk_history,
            "risk_count": count + 1,
            "latest_risk_speaker": role,
            "risk_assessments": risk_assessments,
            last_argument_keys[role]: last_argument,
        },
    )
    return {
        **next_state,
        "risk_result": build_risk_result_from_state(next_state),
    }


def should_continue_risk_analysis(state: ResearchReportGraphState) -> str:
    """根据风险辩论次数决定下一位风险分析员，或进入 Portfolio Manager。"""
    max_rounds = max(1, int(state.get("max_risk_discuss_rounds", 1)))
    if int(state.get("risk_count", 0)) >= 3 * max_rounds:
        return "portfolio_manager"
    latest_speaker = state.get("latest_risk_speaker")
    if latest_speaker == "aggressive":
        return "conservative_risk_analyst"
    if latest_speaker == "conservative":
        return "neutral_risk_analyst"
    return "aggressive_risk_analyst"


def portfolio_manager_node(state: ResearchReportGraphState) -> ResearchReportGraphState:
    """Portfolio Manager 节点。"""
    emit_progress("正在由组合经理生成最终评级、交易信号和风控护栏。")
    final_state = state["prompt_result"].final_state
    client = get_state_client(state)
    config = state.get("config") or ResearchInputConfig()
    past_context = ""
    reflection_summary: dict[str, Any] = {}

    if state.get("enable_memory", True):
        memory_log = TradingMemoryLog(log_path=state.get("memory_log_path"))
        reflection = Reflector(client).resolve_and_load_context(
            memory_log=memory_log,
            symbol=final_state["symbol"],
            holding_days=config.memory_holding_days,
            benchmark_symbol=config.benchmark_symbol,
            benchmark_name=config.benchmark_name,
            resolve_all_pending=config.resolve_all_pending_memory,
        )
        past_context = reflection.past_context
        reflection_summary = {
            "benchmark_symbol": reflection.benchmark.symbol,
            "benchmark_name": reflection.benchmark.name,
            "holding_days": reflection.holding_days,
            "updated_pending_count": reflection.updated_count,
            "resolve_all_pending": config.resolve_all_pending_memory,
        }

    context = build_portfolio_manager_context(
        symbol=final_state["symbol"],
        trade_date=final_state["trade_date"],
        investment_plan=state["investment_plan"],
        trader_plan=state["trader_plan"],
        risk_debate_history=state["risk_result"].risk_history,
        past_context=past_context,
    )
    messages = build_agent_messages("Portfolio Manager", context.prompt)
    structured_result = call_structured_output(
        llm_client=client,
        messages=messages,
        schema_model=PortfolioDecision,
        fallback_factory=build_fallback_portfolio_decision,
        temperature=0.0,
        max_retries=2,
    )
    portfolio_decision = structured_result.value
    final_trade_decision = render_portfolio_decision(portfolio_decision)
    trade_signal = process_portfolio_rating(portfolio_decision.rating.value)
    risk_guardrail = build_risk_guardrail_decision(
        trade_signal=trade_signal,
        trader_action=state["trader_proposal"].action.value,
        trader_position_sizing=state["trader_proposal"].position_sizing,
        trader_stop_loss=state["trader_proposal"].stop_loss,
        risk_assessments=state["risk_result"].risk_assessments,
    )

    if state.get("enable_memory", True):
        memory_log.store_decision(
            symbol=final_state["symbol"],
            trade_date=final_state["trade_date"],
            rating=portfolio_decision.rating.value,
            final_trade_decision=build_memory_decision_text(
                final_trade_decision,
                risk_guardrail,
            ),
        )

    return append_agent_result(
        state,
        "portfolio_manager",
        structured_result.messages,
        {
            "portfolio_decision": portfolio_decision,
            "final_trade_decision": final_trade_decision,
            "trade_signal": trade_signal,
            "risk_guardrail": risk_guardrail,
            "reflection_summary": reflection_summary,
        },
    )


def append_agent_result(
    state: ResearchReportGraphState,
    agent_key: str,
    messages: list[dict[str, Any]],
    extra: dict[str, Any],
    tool_result: Any | None = None,
) -> ResearchReportGraphState:
    """写入单个 Agent 的输出和调试消息。"""
    messages_by_agent = dict(state.get("messages_by_agent", {}))
    messages_by_agent[agent_key] = messages
    tool_call_trace = list(state.get("tool_call_trace", []))
    tool_call_stats = dict(state.get("tool_call_stats") or build_empty_tool_call_stats())

    if tool_result is not None:
        tool_call_trace, tool_call_stats = append_tool_result_trace(
            agent_key=agent_key,
            tool_result=tool_result,
            current_trace=tool_call_trace,
            current_stats=tool_call_stats,
        )

    return {
        **state,
        **extra,
        "messages_by_agent": messages_by_agent,
        "tool_call_trace": tool_call_trace,
        "tool_call_stats": tool_call_stats,
    }


def merge_messages_by_agent(
    state: ResearchReportGraphState,
    new_messages: dict[str, list[dict[str, Any]]],
    extra: dict[str, Any],
) -> ResearchReportGraphState:
    """合并辩论类节点产生的多组调试消息。"""
    messages_by_agent = dict(state.get("messages_by_agent", {}))
    messages_by_agent.update(new_messages)
    return {
        **state,
        **extra,
        "messages_by_agent": messages_by_agent,
    }


def build_empty_tool_call_stats() -> dict[str, Any]:
    """构造空工具调用统计。"""
    return {
        "total_tool_calls": 0,
        "by_agent": {},
    }


def append_tool_result_trace(
    agent_key: str,
    tool_result: Any,
    current_trace: list[dict[str, Any]],
    current_stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把某个 Analyst 的工具调用轨迹合并进全局 state。"""
    agent_trace = [
        {
            **event,
            "agent": agent_key,
        }
        for event in getattr(tool_result, "tool_trace", []) or []
    ]
    trace = current_trace + agent_trace

    by_agent = dict(current_stats.get("by_agent", {}))
    tool_names = sorted(
        {
            str(event.get("tool_name"))
            for event in agent_trace
            if event.get("event") == "assistant_tool_call" and event.get("tool_name")
        }
    )
    by_agent[agent_key] = {
        "tool_call_count": int(getattr(tool_result, "tool_call_count", 0) or 0),
        "tool_names": tool_names,
    }
    total_tool_calls = sum(
        int(item.get("tool_call_count", 0) or 0)
        for item in by_agent.values()
    )
    return trace, {
        "total_tool_calls": total_tool_calls,
        "by_agent": by_agent,
    }


ANALYST_NODE_SPECS = {
    "market": AnalystNodeSpec("market", "market_agent", market_agent_node),
    "sentiment": AnalystNodeSpec("sentiment", "sentiment_agent", sentiment_agent_node),
    "news": AnalystNodeSpec("news", "news_agent", news_agent_node),
    "fundamentals": AnalystNodeSpec(
        "fundamentals",
        "fundamentals_agent",
        fundamentals_agent_node,
    ),
}


def build_research_report_state_graph(
    checkpointer: Any | None = None,
    selected_analysts: str | tuple[str, ...] | list[str] | None = None,
):
    """构建完整研究报告 StateGraph。

    selected_analysts 控制前半段接入哪些 Analyst 节点。
    后半段 Summary、Debate、Research Manager、Trader、Risk、Portfolio Manager
    仍然保持完整，因为它们负责把分析材料转换成最终交易决策。
    """
    analysts = normalize_selected_analysts(selected_analysts)
    setup = GraphSetup(
        state_schema=ResearchReportGraphState,
        prepare_node=prepare_prompts_node,
        analyst_specs=ANALYST_NODE_SPECS,
        summary_node=summary_agent_node,
        bull_node=bull_researcher_node,
        bear_node=bear_researcher_node,
        research_manager_node=research_manager_node,
        trader_node=trader_node,
        aggressive_risk_node=aggressive_risk_analyst_node,
        conservative_risk_node=conservative_risk_analyst_node,
        neutral_risk_node=neutral_risk_analyst_node,
        portfolio_manager_node=portfolio_manager_node,
        should_continue_debate=should_continue_debate,
        should_continue_risk_analysis=should_continue_risk_analysis,
    )
    return setup.setup_graph(
        selected_analysts=analysts,
        checkpointer=checkpointer or create_memory_checkpointer(),
    )


def run_research_report_state_graph(
    symbol: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    enable_memory: bool = True,
    memory_log_path: str | None = None,
    thread_id: str | None = None,
    checkpoint_db_path: str | None = None,
    resume: bool = False,
    use_sqlite_checkpoint: bool = True,
    selected_analysts: str | tuple[str, ...] | list[str] | None = None,
) -> ResearchReportPipelineResult:
    """运行完整 StateGraph，并返回和旧流水线兼容的结果对象。"""
    actual_config = config or ResearchInputConfig()
    configured_analysts = normalize_selected_analysts(
        selected_analysts or actual_config.selected_analysts
    )
    configured_analysts = filter_sentiment_analyst(
        configured_analysts,
        include_sentiment=actual_config.include_sentiment,
    )
    actual_thread_id = thread_id or f"research-{symbol}-{trade_date or 'today'}"
    propagator = ResearchGraphPropagator()
    initial_state = propagator.create_initial_state(
        symbol=symbol,
        trade_date=trade_date,
        config=actual_config,
        llm_client=llm_client,
        temperature=temperature,
        max_debate_rounds=max_debate_rounds,
        max_risk_discuss_rounds=max_risk_discuss_rounds,
        enable_memory=enable_memory,
        memory_log_path=memory_log_path,
        selected_analysts=configured_analysts,
    )

    if not use_sqlite_checkpoint:
        app = build_research_report_state_graph(selected_analysts=configured_analysts)
        output = app.invoke(
            initial_state,
            config=propagator.build_invoke_config(actual_thread_id),
        )
        return finalize_pipeline_result(output, actual_config)

    with create_sqlite_checkpointer(checkpoint_db_path) as checkpointer:
        app = build_research_report_state_graph(
            checkpointer=checkpointer,
            selected_analysts=configured_analysts,
        )
        input_state = propagator.select_input_state(
            initial_state=initial_state,
            thread_id=actual_thread_id,
            resume=resume,
            checkpoint_db_path=checkpoint_db_path,
        )
        output = app.invoke(
            input_state,
            config=propagator.build_invoke_config(actual_thread_id),
        )
        return finalize_pipeline_result(output, actual_config)


def should_resume_thread(
    thread_id: str,
    resume: bool,
    checkpoint_db_path: str | None = None,
) -> bool:
    """判断本次运行是否应该从已有 checkpoint 继续。"""
    if not resume:
        return False
    return has_checkpoint_for_thread(thread_id, checkpoint_db_path)


def finalize_pipeline_result(
    state: ResearchReportGraphState,
    config: ResearchInputConfig,
) -> ResearchReportPipelineResult:
    """把最终 state 转成结果对象，并按配置保存模拟盘和 full_state.json。"""
    result = state_to_pipeline_result(state)
    if config.enable_paper_trading:
        emit_progress("正在根据最终信号写入本地模拟盘。")
        try:
            result.paper_trading_result = run_paper_trading_from_result(
                result=result,
                config=PaperTradingConfig.from_research_config(config),
                llm_client=state.get("llm_client"),
            )
        except Exception as error:
            result.paper_trading_result = {
                "enabled": True,
                "status": "error",
                "error": str(error),
            }
    if config.save_full_state:
        emit_progress("正在保存完整运行状态 full_state.json。")
        result.full_state_log_path = save_full_state_json(
            result,
            output_dir=config.full_state_output_dir,
        )
    return result


def state_to_pipeline_result(state: ResearchReportGraphState) -> ResearchReportPipelineResult:
    """把 StateGraph 最终 state 转成旧代码兼容的 Result。"""
    debate_result = state["debate_result"]
    risk_result = state["risk_result"]
    return ResearchReportPipelineResult(
        prompt_result=state["prompt_result"],
        market_report=get_report_or_missing(state, "market_report", "Market Agent"),
        news_report=get_report_or_missing(state, "news_report", "News Agent"),
        fundamentals_report=get_report_or_missing(
            state,
            "fundamentals_report",
            "Fundamentals Agent",
        ),
        summary_report=state["summary_report"],
        bull_argument=debate_result.last_bull_argument,
        bear_argument=debate_result.last_bear_argument,
        debate_history=debate_result.debate_history,
        max_debate_rounds=state.get("max_debate_rounds", 1),
        research_plan=state["research_plan"],
        investment_plan=state["investment_plan"],
        trader_proposal=state["trader_proposal"],
        trader_plan=state["trader_plan"],
        risk_debate_history=risk_result.risk_history,
        aggressive_risk_argument=risk_result.last_aggressive_argument,
        conservative_risk_argument=risk_result.last_conservative_argument,
        neutral_risk_argument=risk_result.last_neutral_argument,
        portfolio_decision=state["portfolio_decision"],
        final_trade_decision=state["final_trade_decision"],
        trade_signal=state["trade_signal"],
        risk_guardrail=state["risk_guardrail"],
        messages_by_agent=state["messages_by_agent"],
        sentiment_report=get_report_or_missing(
            state,
            "sentiment_report",
            "Sentiment Agent",
        ),
        selected_analysts=tuple(state.get("selected_analysts", ())),
        tool_call_trace=list(state.get("tool_call_trace", [])),
        tool_call_stats=dict(state.get("tool_call_stats", {})),
        reflection_summary=dict(state.get("reflection_summary", {})),
        paper_trading_result=dict(state.get("paper_trading_result", {})),
    )
