"""用户自然语言问题执行入口。

这个文件是“人使用系统”的入口层。

它接收：

    用户自然语言问题

例如：

    大唐发电行情怎么样
    帮我看看京东方A能不能买

然后执行：

    1. route_user_question(...)
       把自然语言转换成结构化路由。

    2. 如果是 single_stock_analysis：
       调用完整 TradingAgents 链路。

    3. 保存最终 Markdown 报告。

    4. 返回一段适合直接给人看的简短回答。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tradingagents_cn.dataflows.market_overview import (
    get_market_overview,
    render_market_overview_text,
)
from tradingagents_cn.dataflows.stock_screening import (
    StockScreeningConfig,
    get_stock_screening_candidates,
    render_stock_screening_text,
)
from tradingagents_cn.agents import (
    build_market_overview_agent_context,
    build_stock_screening_agent_context,
)
from tradingagents_cn.graph.final_report import (
    build_operation_sentence,
    build_trading_status,
    render_paper_trading_brief,
    translate_portfolio_rating,
    translate_trader_action,
    save_final_markdown_report,
)
from tradingagents_cn.graph.display_text import translate_machine_action
from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.graph.progress import emit_progress
from tradingagents_cn.graph.research_report_pipeline import (
    ChatModelClient,
    ResearchReportPipelineResult,
)
from tradingagents_cn.graph.research_report_state_graph import run_research_report_state_graph
from tradingagents_cn.graph.stock_screening_deep_pipeline import (
    render_deep_screening_result,
    run_deep_stock_screening,
)
from tradingagents_cn.llm.deepseek_client import (
    extract_assistant_message,
)
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.router import (
    StockRouteItem,
    UserQuestionIntent,
    UserQuestionRoute,
    route_user_question,
    route_user_question_with_llm,
)


@dataclass
class MultiStockPipelineItem:
    """多股分析中的单只股票运行结果。"""

    route: UserQuestionRoute
    answer: str
    thread_id: str
    report_path: Path
    research_result: ResearchReportPipelineResult


@dataclass
class UserQuestionPipelineResult:
    """用户问题执行结果。"""

    route: UserQuestionRoute
    answer: str
    thread_id: str | None = None
    report_path: Path | None = None
    research_result: ResearchReportPipelineResult | None = None
    multi_results: tuple[MultiStockPipelineItem, ...] = ()


def run_user_question_pipeline(
    question: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    output_dir: str | Path = "outputs",
    use_llm_router: bool = True,
    enable_deep_stock_screening: bool = True,
    deep_stock_screening_top_n: int = 3,
    thread_id: str | None = None,
    resume: bool = False,
    checkpoint_db_path: str | None = None,
) -> UserQuestionPipelineResult:
    """运行用户自然语言问题。

    当前已支持：
        single_stock_analysis

    当前暂未完整支持：
        market_overview
        stock_screening

    为什么先支持单股？
        因为前面已经完成的是“单只股票完整分析发动机”。
        市场概览和筛股需要新的数据层，例如指数、板块、涨跌家数、
        成交额排行、行业热点等，后续再接。
    """
    if use_llm_router:
        emit_progress("正在识别你的问题和股票代码，请稍等。")
        route = route_user_question_with_llm(
            question,
            llm_client=llm_client,
            temperature=0.0,
        )
    else:
        emit_progress("正在用规则识别你的问题和股票代码，请稍等。")
        route = route_user_question(question)

    if route.intent == UserQuestionIntent.SINGLE_STOCK_ANALYSIS:
        if route.symbol is None:
            raise ValueError("路由结果是单股分析，但没有股票代码。")

        actual_trade_date = trade_date or date.today().strftime("%Y-%m-%d")
        actual_thread_id = thread_id or f"single-stock-{route.symbol}-{actual_trade_date}"
        stock_label = route.stock_name or route.symbol
        emit_progress(f"已识别为单股分析：{stock_label}（{route.symbol}）。开始完整分析链路。")
        research_result = run_research_report_state_graph(
            symbol=route.symbol,
            trade_date=actual_trade_date,
            config=config,
            llm_client=llm_client,
            temperature=temperature,
            max_debate_rounds=max_debate_rounds,
            max_risk_discuss_rounds=max_risk_discuss_rounds,
            thread_id=actual_thread_id,
            checkpoint_db_path=checkpoint_db_path,
            resume=resume,
        )

        emit_progress("正在生成 Markdown 报告和终端摘要。")
        report_path = build_report_path(
            output_dir=output_dir,
            symbol=route.symbol,
            stock_name=route.stock_name,
            trade_date=actual_trade_date,
        )
        save_final_markdown_report(research_result, report_path)

        answer = build_single_stock_answer(route, research_result, report_path)
        return UserQuestionPipelineResult(
            route=route,
            answer=answer,
            thread_id=actual_thread_id,
            report_path=report_path,
            research_result=research_result,
        )

    if route.intent == UserQuestionIntent.MULTI_STOCK_ANALYSIS:
        if not route.stock_items:
            raise ValueError("路由结果是多股分析，但没有股票列表。")

        actual_trade_date = trade_date or date.today().strftime("%Y-%m-%d")
        actual_thread_prefix = thread_id or f"multi-stock-{actual_trade_date}"
        total_count = len(route.stock_items)
        stock_text = "、".join(f"{item.stock_name}（{item.symbol}）" for item in route.stock_items)
        emit_progress(f"已识别为多股分析，共 {total_count} 只：{stock_text}。")
        emit_progress("将按顺序逐只运行完整 A 股多智能体分析链路。")

        multi_results: list[MultiStockPipelineItem] = []
        for index, stock_item in enumerate(route.stock_items, start=1):
            single_route = build_single_route_from_stock_item(route.original_question, stock_item)
            item_thread_id = f"{actual_thread_prefix}-{index:02d}-{stock_item.symbol}"
            emit_progress(
                f"开始分析第 {index}/{total_count} 只："
                f"{stock_item.stock_name}（{stock_item.symbol}）。"
            )
            research_result = run_research_report_state_graph(
                symbol=stock_item.symbol,
                trade_date=actual_trade_date,
                config=config,
                llm_client=llm_client,
                temperature=temperature,
                max_debate_rounds=max_debate_rounds,
                max_risk_discuss_rounds=max_risk_discuss_rounds,
                thread_id=item_thread_id,
                checkpoint_db_path=checkpoint_db_path,
                resume=resume,
            )

            emit_progress(
                f"正在保存第 {index}/{total_count} 只报告："
                f"{stock_item.stock_name}（{stock_item.symbol}）。"
            )
            report_path = build_report_path(
                output_dir=output_dir,
                symbol=stock_item.symbol,
                stock_name=stock_item.stock_name,
                trade_date=actual_trade_date,
            )
            save_final_markdown_report(research_result, report_path)
            item_answer = build_single_stock_answer(single_route, research_result, report_path)
            multi_results.append(
                MultiStockPipelineItem(
                    route=single_route,
                    answer=item_answer,
                    thread_id=item_thread_id,
                    report_path=report_path,
                    research_result=research_result,
                )
            )

        answer = build_multi_stock_answer(
            route=route,
            multi_results=tuple(multi_results),
            thread_prefix=actual_thread_prefix,
        )
        return UserQuestionPipelineResult(
            route=route,
            answer=answer,
            thread_id=actual_thread_prefix,
            multi_results=tuple(multi_results),
        )

    if route.intent == UserQuestionIntent.MARKET_OVERVIEW:
        return UserQuestionPipelineResult(
            route=route,
            answer=run_market_overview_answer(
                question=question,
                llm_client=llm_client,
                temperature=temperature,
            ),
            thread_id=None,
        )

    if route.intent == UserQuestionIntent.STOCK_SCREENING:
        return UserQuestionPipelineResult(
            route=route,
            answer=run_stock_screening_answer(
                question=question,
                trade_date=trade_date,
                config=config,
                llm_client=llm_client,
                temperature=temperature,
                deep_screening=enable_deep_stock_screening,
                deep_top_n=deep_stock_screening_top_n,
                max_debate_rounds=max_debate_rounds,
                max_risk_discuss_rounds=max_risk_discuss_rounds,
            ),
            thread_id=None,
        )

    if route.intent == UserQuestionIntent.OUT_OF_SCOPE:
        return UserQuestionPipelineResult(
            route=route,
            answer=(
                "这个问题和 A 股、股票市场或投资研究无关，"
                "当前系统只处理 A 股投研相关问题，所以不会继续调用分析链路。"
            ),
            thread_id=None,
        )

    return UserQuestionPipelineResult(
        route=route,
        answer=f"暂时没有识别出可执行的分析任务。原因：{route.reason}",
        thread_id=None,
    )


def build_report_path(
    output_dir: str | Path,
    symbol: str,
    stock_name: str | None,
    trade_date: str,
) -> Path:
    """构造最终报告保存路径。

    文件名里放股票名，是为了让人打开 outputs 目录时一眼能看懂。
    同时保留股票代码，是为了避免不同股票简称相似时不好区分。

    示例：
        京东方A_000725_2026-06-18_final_report.md
    """
    name_part = sanitize_filename_part(stock_name or symbol)
    symbol_part = sanitize_filename_part(symbol)
    return Path(output_dir) / f"{name_part}_{symbol_part}_{trade_date}_final_report.md"


def run_market_overview_answer(
    question: str,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
) -> str:
    """运行市场概览回答。

    当前流程：
        1. 获取市场概览原材料；
        2. 构造 Market Overview Agent Prompt；
        3. 调用大模型生成中文市场报告；
        4. 如果模型失败，返回原材料和错误提示。
    """
    overview = get_market_overview()
    materials = render_market_overview_text(overview)
    context = build_market_overview_agent_context(
        question=question,
        materials=materials,
    )
    client = llm_client or create_chat_client()

    try:
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": context.prompt,
                }
            ],
            temperature=temperature,
        )
        message = extract_assistant_message(response)
        content = str(message.get("content") or "").strip()
        if content:
            return content
        return f"市场概览模型返回为空，以下为原始市场材料：\n\n{materials}"
    except Exception as error:
        return f"市场概览模型调用失败，以下为原始市场材料。\n错误：{error}\n\n{materials}"


def run_stock_screening_answer(
    question: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
    deep_screening: bool = False,
    deep_top_n: int = 3,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
) -> str:
    """运行选股/筛股回答。

    当前流程：
        1. 获取全市场行情候选池；
        2. 构造 Stock Screening Agent Prompt；
        3. 调用大模型生成候选观察名单；
        4. 如果 deep_screening=True，把前 N 只候选股继续送进完整单股链路；
        5. 如果模型失败，返回原始候选池材料。
    """
    candidates = get_stock_screening_candidates(StockScreeningConfig())
    materials = render_stock_screening_text(candidates)

    if deep_screening:
        deep_result = run_deep_stock_screening(
            candidates=candidates,
            trade_date=trade_date,
            top_n=deep_top_n,
            config=config,
            llm_client=llm_client,
            temperature=temperature,
            max_debate_rounds=max_debate_rounds,
            max_risk_discuss_rounds=max_risk_discuss_rounds,
        )
        deep_report = render_deep_screening_result(deep_result)
        return "\n\n".join(
            [
                "已完成候选股深度分析。下面的排序不是只看初筛强度，"
                "而是把候选股逐个运行完整单股 TradingAgents 链路后的结果。",
                deep_report,
            ]
        )

    context = build_stock_screening_agent_context(
        question=question,
        materials=materials,
    )
    client = llm_client or create_chat_client()

    try:
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": context.prompt,
                }
            ],
            temperature=temperature,
        )
        message = extract_assistant_message(response)
        content = str(message.get("content") or "").strip()
        if content:
            return content
        return f"选股模型返回为空，以下为原始候选池材料：\n\n{materials}"
    except Exception as error:
        return f"选股模型调用失败，以下为原始候选池材料。\n错误：{error}\n\n{materials}"


def sanitize_filename_part(text: str) -> str:
    """清理 Windows 文件名不能使用的字符。

    Windows 文件名不能包含这些符号：
        小于号、大于号、冒号、双引号、斜杠、反斜杠、竖线、问号、星号。

    股票名一般不会包含这些字符，
    但用户问题或模型识别结果不一定完全干净，
    所以这里统一做一次兜底。
    """
    invalid_chars = '<>:"/\\|?*'
    cleaned = str(text or "").strip()
    for char in invalid_chars:
        cleaned = cleaned.replace(char, "_")
    return cleaned or "未知股票"


def build_single_route_from_stock_item(
    original_question: str,
    stock_item: StockRouteItem,
) -> UserQuestionRoute:
    """把多股列表中的一项转换成单股路由。

    后面的完整分析链路仍然是单股链路。
    多股执行只是把多只股票排队逐只送进去。
    """
    return UserQuestionRoute(
        intent=UserQuestionIntent.SINGLE_STOCK_ANALYSIS,
        original_question=original_question,
        symbol=stock_item.symbol,
        stock_name=stock_item.stock_name,
        analysis_depth="full",
        reason=f"多股问题拆分出的单股分析项，来源：{stock_item.match_source}。",
    )


def build_single_stock_answer(
    route: UserQuestionRoute,
    result: ResearchReportPipelineResult,
    report_path: Path,
) -> str:
    """构造给用户看的单股简短回答。"""
    stock_label = route.stock_name or route.symbol or "该股票"
    trader_action = result.trader_proposal.action.value
    portfolio_rating = result.portfolio_decision.rating.value
    research_rating = result.research_plan.recommendation.value
    status = build_trading_status(result)
    position_sizing = result.trader_proposal.position_sizing or "模型没有给出明确仓位比例。"

    return "\n".join(
        [
            f"结论：{status.summary}",
            "",
            f"已完成 {stock_label}（{route.symbol}）的完整 A 股多智能体分析。",
            "",
            f"交易状态：{status.label}",
            f"现在能不能买：{status.current_action}",
            f"如果已经持有：{status.if_holding}",
            f"如果还没持有：{status.if_not_holding}",
            f"等待条件：{status.watch_condition}",
            f"交易员动作：{translate_trader_action(trader_action)}",
            f"组合经理最终评级：{translate_portfolio_rating(portfolio_rating)}",
            f"研究经理评级：{translate_portfolio_rating(research_rating)}",
            f"机器交易信号：{translate_machine_action(result.trade_signal.action)}，{result.trade_signal.chinese_action}",
            f"仓位建议：{position_sizing}",
            f"程序风控护栏：{result.risk_guardrail.chinese_summary}",
            f"模拟盘：{render_paper_trading_brief(result)}",
            "",
            f"核心理由：{result.portfolio_decision.executive_summary}",
            "",
            f"完整报告已保存：{report_path}",
            f"完整状态日志：{result.full_state_log_path or '未保存'}",
        ]
    )


def build_multi_stock_answer(
    route: UserQuestionRoute,
    multi_results: tuple[MultiStockPipelineItem, ...],
    thread_prefix: str,
) -> str:
    """构造多股问题的终端汇总回答。"""
    if not multi_results:
        return "识别到多股问题，但没有任何股票完成分析。"

    lines = [
        f"已完成 {len(multi_results)} 只股票的逐只完整 A 股多智能体分析。",
        "",
        "说明：这是逐只分析汇总，不是按优先级排序，也不是自动推荐买入列表。",
        f"多股任务 thread_id 前缀：{thread_prefix}",
        "",
        "========== 多股结论汇总 ==========",
    ]

    for index, item in enumerate(multi_results, start=1):
        result = item.research_result
        trader_action = result.trader_proposal.action.value
        portfolio_rating = result.portfolio_decision.rating.value
        status = build_trading_status(result)
        position_sizing = result.trader_proposal.position_sizing or "模型没有给出明确仓位比例。"
        lines.extend(
            [
                "",
                f"{index}. {item.route.stock_name}（{item.route.symbol}）",
                f"   操作结论：{status.summary}",
                f"   交易状态：{status.label}",
                f"   现在能不能买：{status.current_action}",
                f"   交易员动作：{translate_trader_action(trader_action)}",
                f"   组合经理评级：{translate_portfolio_rating(portfolio_rating)}",
                f"   机器交易信号：{translate_machine_action(result.trade_signal.action)}，{result.trade_signal.chinese_action}",
                f"   仓位建议：{position_sizing}",
                f"   核心理由：{result.portfolio_decision.executive_summary}",
                f"   报告：{item.report_path}",
                f"   thread_id：{item.thread_id}",
            ]
        )

    lines.extend(
        [
            "",
            "完整报告已分别保存到上面列出的 Markdown 文件。",
        ]
    )
    return "\n".join(lines)
