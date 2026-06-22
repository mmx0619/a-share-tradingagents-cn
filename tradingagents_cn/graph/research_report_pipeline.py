"""A 股研究报告生成流程。

上一层 research_pipeline.py 做到的是：

    真实数据 -> LangGraph -> 三个 Agent Prompt

这个文件继续往后走一步：

    三个 Agent Prompt -> 调用真实大模型 -> 三份 Agent 报告 -> 综合研究结论 -> 多空辩论

当前包含三个分析 Agent：

    Market Agent
        根据行情和技术指标，生成技术面分析报告。

    News Agent
        根据新闻材料，生成消息面分析报告。

    Fundamentals Agent
        根据公司资料和财务报表，生成基本面分析报告。

    Summary Agent
        汇总前三份报告，生成综合研究结论。

    Bull / Bear Researcher
        基于研究报告进行多空辩论。

重要说明：
    这一步仍然不是最终交易决策。
    它只是让三个分析员各自写报告，让综合研究员做汇总，
    再让多头和空头研究员按配置轮数进行辩论。

后续还会继续接：
    - Bull / Bear Researcher：多空辩论；
    - Risk Control：风险控制；
    - Trader：交易计划；
    - Portfolio Manager：最终组合决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tradingagents_cn.agents import (
    DebateArgument,
    build_bear_researcher_context,
    build_bull_researcher_context,
    build_aggressive_risk_context,
    build_conservative_risk_context,
    build_fallback_debate_argument,
    build_fallback_risk_assessment,
    build_neutral_risk_context,
    build_portfolio_manager_context,
    build_research_manager_context,
    build_summary_agent_context,
    build_trader_agent_context,
    parse_research_plan_from_text,
    parse_portfolio_decision_from_text,
    parse_trader_proposal_from_text,
    render_portfolio_decision,
    render_debate_argument,
    render_research_plan,
    render_risk_assessment,
    render_trader_proposal,
    ResearchPlan,
    PortfolioDecision,
    RiskAssessment,
    TraderProposal,
)
from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.graph.research_pipeline import (
    ResearchPromptPipelineResult,
    run_research_prompt_pipeline,
)
from tradingagents_cn.graph.research_workflow import ResearchWorkflowState
from tradingagents_cn.graph.signal_processing import (
    RiskGuardrailDecision,
    TradeSignal,
    build_risk_guardrail_decision,
    process_portfolio_rating,
    render_risk_guardrail_decision,
)
from tradingagents_cn.graph.run_state_logging import save_full_state_json
from tradingagents_cn.llm.deepseek_client import (
    extract_assistant_message,
)
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.llm.structured_output import call_structured_output
from tradingagents_cn.memory import TradingMemoryLog
from tradingagents_cn.paper_trading import (
    PaperTradingConfig,
    run_paper_trading_from_result,
)


class ChatModelClient(Protocol):
    """聊天模型客户端协议。

    为什么这里不直接写死 DeepSeek？

        因为你的目标不是只跑 DeepSeek。
        后续你还可能接 OpenAI、Gemini、Kimi 等模型。

    只要某个客户端实现了 chat(...) 方法，
    并且返回 OpenAI / DeepSeek 类似的 JSON 结构，
    就可以放进这个流程里使用。
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """发送消息给聊天模型，并返回原始 JSON。"""


@dataclass
class ResearchReportPipelineResult:
    """研究报告流程的返回结果。

    prompt_result:
        上一步 Prompt 流程的完整结果。
        如果你想调试“到底发给大模型的 Prompt 是什么”，看这里。

    market_report:
        Market Agent 生成的技术面报告。

    news_report:
        News Agent 生成的新闻面报告。

    fundamentals_report:
        Fundamentals Agent 生成的基本面报告。

    summary_report:
        Summary Agent 生成的综合研究结论。

    bull_argument:
        Bull Researcher 最后一轮生成的多头论点。

    bear_argument:
        Bear Researcher 最后一轮生成的空头论点。

    debate_history:
        当前已经发生的多空辩论记录。

    max_debate_rounds:
        本次运行配置的多空辩论轮数。
        一轮等于：
            Bull 发言一次；
            Bear 发言一次。

    research_plan:
        Research Manager 输出的结构化研究计划对象。

    investment_plan:
        Research Manager 输出的 Markdown 研究计划。
        后续 Trader Agent 会读取这个字段。

    trader_proposal:
        Trader Agent 输出的结构化交易提案。

    trader_plan:
        Trader Agent 输出的 Markdown 交易计划。

    risk_debate_history:
        三位风险分析员的完整风险辩论历史。

    aggressive_risk_argument / conservative_risk_argument / neutral_risk_argument:
        三位风险分析员最后一轮观点。

    portfolio_decision:
        Portfolio Manager 输出的结构化最终组合决策。

    final_trade_decision:
        Portfolio Manager 输出的 Markdown 最终交易决策。

    trade_signal:
        从 Portfolio Manager 评级确定性转换出来的机器可读信号。

    risk_guardrail:
        程序层面的风控护栏。
        它根据最终信号、Trader 仓位建议和 Risk Debate 结果，
        计算是否允许新增仓位、目标仓位上限和单次加仓上限。

    messages_by_agent:
        每个 Agent 的完整消息列表。
        这对调试非常有用：
            可以看到 system/user/assistant 每一步内容。
    sentiment_report:
        独立 Sentiment Analyst 的情绪面报告。
        老的普通 Prompt 流程暂时可以为空，StateGraph 正式流程会生成它。
    selected_analysts:
        本次主图实际启用的 Analyst 列表。
    tool_call_trace / tool_call_stats:
        Tool Calling 的机器可读轨迹和统计。
    reflection_summary:
        记忆复盘本轮更新了多少 pending 记录、使用哪个 A 股基准。
    paper_trading_result:
        模拟盘自动交易结果。
        默认是空字典；只有 ResearchInputConfig.enable_paper_trading=True 时才会生成订单。
    full_state_log_path:
        full_state.json 的保存路径。
    """

    prompt_result: ResearchPromptPipelineResult
    market_report: str
    news_report: str
    fundamentals_report: str
    summary_report: str
    bull_argument: str
    bear_argument: str
    debate_history: str
    max_debate_rounds: int
    research_plan: ResearchPlan
    investment_plan: str
    trader_proposal: TraderProposal
    trader_plan: str
    risk_debate_history: str
    aggressive_risk_argument: str
    conservative_risk_argument: str
    neutral_risk_argument: str
    portfolio_decision: PortfolioDecision
    final_trade_decision: str
    trade_signal: TradeSignal
    risk_guardrail: RiskGuardrailDecision
    messages_by_agent: dict[str, list[dict[str, Any]]]
    sentiment_report: str = ""
    selected_analysts: tuple[str, ...] = field(default_factory=tuple)
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_call_stats: dict[str, Any] = field(default_factory=dict)
    reflection_summary: dict[str, Any] = field(default_factory=dict)
    paper_trading_result: dict[str, Any] = field(default_factory=dict)
    full_state_log_path: Path | None = None

    @property
    def final_state(self) -> ResearchWorkflowState:
        """返回 LangGraph 执行完成后的 final_state。"""
        return self.prompt_result.final_state

    @property
    def data_errors(self) -> list[str]:
        """返回数据准备阶段的非致命错误。"""
        return self.prompt_result.data_errors


def run_research_report_pipeline(
    symbol: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    enable_memory: bool = True,
    memory_log_path: str | None = None,
) -> ResearchReportPipelineResult:
    """运行正式的 A 股研究报告生成流程。

    参数：
        symbol:
            A 股股票代码，例如 600519、002361。

        trade_date:
            分析日期，格式是 YYYY-MM-DD。
            不传时使用电脑当前日期。

        config:
            数据准备配置。
            用它控制是否获取实时行情、新闻、基本面等。

        llm_client:
            大模型客户端。
            不传时默认使用 DeepSeekChatClient，
            也就是读取环境变量 DEEPSEEK_API_KEY。

        temperature:
            模型随机性。
            股票研究场景一般不希望太发散，所以默认 0.2。

        max_debate_rounds:
            多空辩论轮数。

            1 表示：
                Bull 发言一次；
                Bear 发言一次。

            2 表示：
                Bull -> Bear -> Bull -> Bear。

            原版 TradingAgents 也是用 max_debate_rounds 控制，
            并且一轮包含两个研究员各发言一次。

        max_risk_discuss_rounds:
            风险辩论轮数。

            1 表示：
                Aggressive -> Conservative -> Neutral。

            原版 TradingAgents 一轮风险讨论包含三位风险分析员各发言一次。

    执行流程：
        1. 先调用 run_research_prompt_pipeline(...)
           准备真实数据，并通过 LangGraph 生成三个 Agent Prompt。

        2. 把 market_prompt 发给大模型，
           得到 market_report。

        3. 把 news_prompt 发给大模型，
           得到 news_report。

        4. 把 fundamentals_prompt 发给大模型，
           得到 fundamentals_report。

        5. 把前三份报告组合成 Summary Agent Prompt，
           得到 summary_report。

        6. 让 Bull / Bear Researcher 按 max_debate_rounds 进行多空辩论。

        7. 让 Research Manager 基于 debate_history 输出结构化研究计划。

        8. 让 Trader Agent 把 investment_plan 转换成交易提案。

        9. 让 Aggressive / Conservative / Neutral 三位风险分析员审查交易提案。

        10. 让 Portfolio Manager 基于交易提案、风险辩论和历史记忆输出最终决策。

        11. 返回报告、辩论记录、研究计划、交易提案、风险辩论、最终决策和调试消息。

    重要理解：
        Market / News / Fundamentals 是并列分析材料。
        Summary Agent 是对前三份材料的综合汇总。
        Bull / Bear 是辩论角色，仍然不是最终交易决策。
        Research Manager 开始给出明确评级，但仍作为个人投资研究辅助。
        Trader Agent 进一步给出 Buy / Hold / Sell 交易提案。
        Portfolio Manager 输出最终组合评级和最终交易决策。
    """
    actual_config = config or ResearchInputConfig()
    prompt_result = run_research_prompt_pipeline(
        symbol=symbol,
        trade_date=trade_date,
        config=actual_config,
    )

    client = llm_client or create_chat_client()
    if max_debate_rounds < 1:
        raise ValueError("max_debate_rounds 必须大于等于 1。")
    if max_risk_discuss_rounds < 1:
        raise ValueError("max_risk_discuss_rounds 必须大于等于 1。")

    market_messages = build_agent_messages(
        agent_name="Market Agent",
        prompt=prompt_result.market_prompt,
    )
    market_report = call_agent_report(
        llm_client=client,
        messages=market_messages,
        temperature=temperature,
    )

    news_messages = build_agent_messages(
        agent_name="News Agent",
        prompt=prompt_result.news_prompt,
    )
    news_report = call_agent_report(
        llm_client=client,
        messages=news_messages,
        temperature=temperature,
    )

    fundamentals_messages = build_agent_messages(
        agent_name="Fundamentals Agent",
        prompt=prompt_result.fundamentals_prompt,
    )
    fundamentals_report = call_agent_report(
        llm_client=client,
        messages=fundamentals_messages,
        temperature=temperature,
    )

    summary_context = build_summary_agent_context(
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
    )
    summary_messages = build_agent_messages(
        agent_name="Summary Agent",
        prompt=summary_context.prompt,
    )
    summary_report = call_agent_report(
        llm_client=client,
        messages=summary_messages,
        temperature=temperature,
    )

    debate_result = run_bull_bear_debate(
        llm_client=client,
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        summary_report=summary_report,
        max_debate_rounds=max_debate_rounds,
        temperature=temperature,
    )

    research_manager_context = build_research_manager_context(
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        debate_history=debate_result.debate_history,
    )
    research_manager_messages = build_agent_messages(
        agent_name="Research Manager",
        prompt=research_manager_context.prompt,
    )
    research_plan_text = call_agent_report(
        llm_client=client,
        messages=research_manager_messages,
        temperature=temperature,
    )
    research_plan = parse_research_plan_from_text(research_plan_text)
    investment_plan = render_research_plan(research_plan)

    trader_context = build_trader_agent_context(
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
    )
    trader_messages = build_agent_messages(
        agent_name="Trader Agent",
        prompt=trader_context.prompt,
    )
    trader_proposal_text = call_agent_report(
        llm_client=client,
        messages=trader_messages,
        temperature=temperature,
    )
    trader_proposal = parse_trader_proposal_from_text(trader_proposal_text)
    trader_plan = render_trader_proposal(trader_proposal)

    risk_result = run_risk_debate(
        llm_client=client,
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        max_risk_discuss_rounds=max_risk_discuss_rounds,
        temperature=temperature,
    )

    memory_log = TradingMemoryLog(log_path=memory_log_path) if enable_memory else None
    past_context = ""
    if memory_log is not None:
        memory_log.resolve_pending_outcomes(
            prompt_result.final_state["symbol"],
            llm_client=client,
        )
        past_context = memory_log.get_past_context(prompt_result.final_state["symbol"])

    portfolio_context = build_portfolio_manager_context(
        symbol=prompt_result.final_state["symbol"],
        trade_date=prompt_result.final_state["trade_date"],
        investment_plan=investment_plan,
        trader_plan=trader_plan,
        risk_debate_history=risk_result.risk_history,
        past_context=past_context,
    )
    portfolio_messages = build_agent_messages(
        agent_name="Portfolio Manager",
        prompt=portfolio_context.prompt,
    )
    portfolio_decision_text = call_agent_report(
        llm_client=client,
        messages=portfolio_messages,
        temperature=temperature,
    )
    portfolio_decision = parse_portfolio_decision_from_text(portfolio_decision_text)
    final_trade_decision = render_portfolio_decision(portfolio_decision)
    trade_signal = process_portfolio_rating(portfolio_decision.rating.value)
    risk_guardrail = build_risk_guardrail_decision(
        trade_signal=trade_signal,
        trader_action=trader_proposal.action.value,
        trader_position_sizing=trader_proposal.position_sizing,
        trader_stop_loss=trader_proposal.stop_loss,
        risk_assessments=risk_result.risk_assessments,
    )

    if memory_log is not None:
        memory_log.store_decision(
            symbol=prompt_result.final_state["symbol"],
            trade_date=prompt_result.final_state["trade_date"],
            rating=portfolio_decision.rating.value,
            final_trade_decision=build_memory_decision_text(
                final_trade_decision,
                risk_guardrail,
            ),
        )

    result = ResearchReportPipelineResult(
        prompt_result=prompt_result,
        market_report=market_report,
        news_report=news_report,
        fundamentals_report=fundamentals_report,
        summary_report=summary_report,
        bull_argument=debate_result.last_bull_argument,
        bear_argument=debate_result.last_bear_argument,
        debate_history=debate_result.debate_history,
        max_debate_rounds=max_debate_rounds,
        research_plan=research_plan,
        investment_plan=investment_plan,
        trader_proposal=trader_proposal,
        trader_plan=trader_plan,
        risk_debate_history=risk_result.risk_history,
        aggressive_risk_argument=risk_result.last_aggressive_argument,
        conservative_risk_argument=risk_result.last_conservative_argument,
        neutral_risk_argument=risk_result.last_neutral_argument,
        portfolio_decision=portfolio_decision,
        final_trade_decision=final_trade_decision,
        trade_signal=trade_signal,
        risk_guardrail=risk_guardrail,
        messages_by_agent={
            "market": market_messages,
            "news": news_messages,
            "fundamentals": fundamentals_messages,
            "summary": summary_messages,
            **debate_result.messages_by_agent,
            "research_manager": research_manager_messages,
            "trader": trader_messages,
            **risk_result.messages_by_agent,
            "portfolio_manager": portfolio_messages,
        },
        sentiment_report="",
        selected_analysts=("market", "news", "fundamentals"),
    )
    if actual_config.enable_paper_trading:
        try:
            result.paper_trading_result = run_paper_trading_from_result(
                result=result,
                config=PaperTradingConfig.from_research_config(actual_config),
                llm_client=client,
            )
        except Exception as error:
            result.paper_trading_result = {
                "enabled": True,
                "status": "error",
                "error": str(error),
            }
    if actual_config.save_full_state:
        result.full_state_log_path = save_full_state_json(
            result,
            output_dir=actual_config.full_state_output_dir,
        )
    return result


@dataclass
class DebateRunResult:
    """多空辩论运行结果。

    last_bull_argument:
        最后一轮 Bull Researcher 的发言。

    last_bear_argument:
        最后一轮 Bear Researcher 的发言。

    debate_history:
        完整辩论历史。

    messages_by_agent:
        每一轮每个研究员的完整 messages。
        key 的格式是：
            bull_round_1
            bear_round_1
            bull_round_2
            bear_round_2
    """

    last_bull_argument: str
    last_bear_argument: str
    debate_history: str
    messages_by_agent: dict[str, list[dict[str, Any]]]


def run_bull_bear_debate(
    llm_client: ChatModelClient,
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    summary_report: str,
    max_debate_rounds: int = 1,
    temperature: float = 0.2,
) -> DebateRunResult:
    """运行多轮 Bull / Bear 辩论。

    这里复刻原版 TradingAgents 的轮数含义：

        一轮 = Bull 发言一次 + Bear 发言一次。

    例如：

        max_debate_rounds = 1
            Bull -> Bear

        max_debate_rounds = 2
            Bull -> Bear -> Bull -> Bear

    每次发言都会把之前的 debate_history 放进 Prompt，
    并把上一轮对手观点放进 last_bear_argument 或 last_bull_argument。
    """
    if max_debate_rounds < 1:
        raise ValueError("max_debate_rounds 必须大于等于 1。")

    debate_history = ""
    last_bull_argument = ""
    last_bear_argument = ""
    messages_by_agent: dict[str, list[dict[str, Any]]] = {}

    for round_index in range(1, max_debate_rounds + 1):
        bull_context = build_bull_researcher_context(
            symbol=symbol,
            trade_date=trade_date,
            market_report=market_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            summary_report=summary_report,
            debate_history=debate_history,
            last_bear_argument=last_bear_argument,
        )
        bull_messages = build_agent_messages(
            agent_name=f"Bull Researcher 第 {round_index} 轮",
            prompt=bull_context.prompt,
        )
        # 多空辩论会被 Research Manager 读取并裁判。
        # 这里要求 Bull 返回 DebateArgument，避免关键证据、反驳和不确定性丢失。
        bull_result = call_structured_output(
            llm_client=llm_client,
            messages=bull_messages,
            schema_model=DebateArgument,
            fallback_factory=lambda error: build_fallback_debate_argument(
                "bull",
                error,
            ),
            temperature=0.0,
            max_retries=2,
        )
        bull_argument_body = render_debate_argument(bull_result.value)
        last_bull_argument = f"Bull Researcher Round {round_index}: {bull_argument_body}"
        debate_history = append_debate_argument(debate_history, last_bull_argument)
        messages_by_agent[f"bull_round_{round_index}"] = bull_result.messages

        bear_context = build_bear_researcher_context(
            symbol=symbol,
            trade_date=trade_date,
            market_report=market_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            summary_report=summary_report,
            debate_history=debate_history,
            last_bull_argument=last_bull_argument,
        )
        bear_messages = build_agent_messages(
            agent_name=f"Bear Researcher 第 {round_index} 轮",
            prompt=bear_context.prompt,
        )
        bear_result = call_structured_output(
            llm_client=llm_client,
            messages=bear_messages,
            schema_model=DebateArgument,
            fallback_factory=lambda error: build_fallback_debate_argument(
                "bear",
                error,
            ),
            temperature=0.0,
            max_retries=2,
        )
        bear_argument_body = render_debate_argument(bear_result.value)
        last_bear_argument = f"Bear Researcher Round {round_index}: {bear_argument_body}"
        debate_history = append_debate_argument(debate_history, last_bear_argument)
        messages_by_agent[f"bear_round_{round_index}"] = bear_result.messages

    return DebateRunResult(
        last_bull_argument=last_bull_argument,
        last_bear_argument=last_bear_argument,
        debate_history=debate_history,
        messages_by_agent=messages_by_agent,
    )


def append_debate_argument(debate_history: str, argument: str) -> str:
    """把一段新辩论发言追加到历史里。"""
    if not debate_history:
        return argument
    return f"{debate_history}\n\n{argument}"


def build_memory_decision_text(
    final_trade_decision: str,
    risk_guardrail: RiskGuardrailDecision,
) -> str:
    """构造写入交易记忆的最终决策文本。"""
    return "\n\n".join(
        [
            final_trade_decision,
            "## 程序风控护栏",
            render_risk_guardrail_decision(risk_guardrail),
        ]
    )


@dataclass
class RiskDebateRunResult:
    """风险辩论运行结果。"""

    last_aggressive_argument: str
    last_conservative_argument: str
    last_neutral_argument: str
    risk_history: str
    risk_assessments: list[RiskAssessment]
    messages_by_agent: dict[str, list[dict[str, Any]]]


def run_risk_debate(
    llm_client: ChatModelClient,
    symbol: str,
    trade_date: str,
    market_report: str,
    news_report: str,
    fundamentals_report: str,
    investment_plan: str,
    trader_plan: str,
    max_risk_discuss_rounds: int = 1,
    temperature: float = 0.2,
) -> RiskDebateRunResult:
    """运行多轮风险辩论。

    对齐原版 TradingAgents 的含义：

        一轮风险讨论 =
            Aggressive 发言一次
            Conservative 发言一次
            Neutral 发言一次

    原版的节点顺序是：

        Aggressive -> Conservative -> Neutral -> Aggressive ...

    达到 3 * max_risk_discuss_rounds 次发言后，
    进入 Portfolio Manager。
    """
    if max_risk_discuss_rounds < 1:
        raise ValueError("max_risk_discuss_rounds 必须大于等于 1。")

    risk_history = ""
    last_aggressive_argument = ""
    last_conservative_argument = ""
    last_neutral_argument = ""
    risk_assessments: list[RiskAssessment] = []
    messages_by_agent: dict[str, list[dict[str, Any]]] = {}

    for round_index in range(1, max_risk_discuss_rounds + 1):
        aggressive_context = build_aggressive_risk_context(
            symbol=symbol,
            trade_date=trade_date,
            market_report=market_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            investment_plan=investment_plan,
            trader_plan=trader_plan,
            risk_history=risk_history,
            last_conservative_argument=last_conservative_argument,
            last_neutral_argument=last_neutral_argument,
        )
        aggressive_messages = build_agent_messages(
            agent_name=f"Aggressive Risk Analyst 第 {round_index} 轮",
            prompt=aggressive_context.prompt,
        )
        # 风险节点会影响最终交易决策，不能只依赖普通文本。
        # 这里要求模型返回 RiskAssessment，字段和值都必须通过 Pydantic 校验。
        aggressive_result = call_structured_output(
            llm_client=llm_client,
            messages=aggressive_messages,
            schema_model=RiskAssessment,
            fallback_factory=lambda error: build_fallback_risk_assessment(
                "aggressive",
                error,
            ),
            temperature=0.0,
            max_retries=2,
        )
        aggressive_body = render_risk_assessment(aggressive_result.value)
        risk_assessments.append(aggressive_result.value)
        last_aggressive_argument = (
            f"Aggressive Risk Analyst Round {round_index}: {aggressive_body}"
        )
        risk_history = append_debate_argument(risk_history, last_aggressive_argument)
        messages_by_agent[f"aggressive_risk_round_{round_index}"] = (
            aggressive_result.messages
        )

        conservative_context = build_conservative_risk_context(
            symbol=symbol,
            trade_date=trade_date,
            market_report=market_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            investment_plan=investment_plan,
            trader_plan=trader_plan,
            risk_history=risk_history,
            last_aggressive_argument=last_aggressive_argument,
            last_neutral_argument=last_neutral_argument,
        )
        conservative_messages = build_agent_messages(
            agent_name=f"Conservative Risk Analyst 第 {round_index} 轮",
            prompt=conservative_context.prompt,
        )
        conservative_result = call_structured_output(
            llm_client=llm_client,
            messages=conservative_messages,
            schema_model=RiskAssessment,
            fallback_factory=lambda error: build_fallback_risk_assessment(
                "conservative",
                error,
            ),
            temperature=0.0,
            max_retries=2,
        )
        conservative_body = render_risk_assessment(conservative_result.value)
        risk_assessments.append(conservative_result.value)
        last_conservative_argument = (
            f"Conservative Risk Analyst Round {round_index}: {conservative_body}"
        )
        risk_history = append_debate_argument(risk_history, last_conservative_argument)
        messages_by_agent[f"conservative_risk_round_{round_index}"] = (
            conservative_result.messages
        )

        neutral_context = build_neutral_risk_context(
            symbol=symbol,
            trade_date=trade_date,
            market_report=market_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            investment_plan=investment_plan,
            trader_plan=trader_plan,
            risk_history=risk_history,
            last_aggressive_argument=last_aggressive_argument,
            last_conservative_argument=last_conservative_argument,
        )
        neutral_messages = build_agent_messages(
            agent_name=f"Neutral Risk Analyst 第 {round_index} 轮",
            prompt=neutral_context.prompt,
        )
        neutral_result = call_structured_output(
            llm_client=llm_client,
            messages=neutral_messages,
            schema_model=RiskAssessment,
            fallback_factory=lambda error: build_fallback_risk_assessment(
                "neutral",
                error,
            ),
            temperature=0.0,
            max_retries=2,
        )
        neutral_body = render_risk_assessment(neutral_result.value)
        risk_assessments.append(neutral_result.value)
        last_neutral_argument = (
            f"Neutral Risk Analyst Round {round_index}: {neutral_body}"
        )
        risk_history = append_debate_argument(risk_history, last_neutral_argument)
        messages_by_agent[f"neutral_risk_round_{round_index}"] = (
            neutral_result.messages
        )

    return RiskDebateRunResult(
        last_aggressive_argument=last_aggressive_argument,
        last_conservative_argument=last_conservative_argument,
        last_neutral_argument=last_neutral_argument,
        risk_history=risk_history,
        risk_assessments=risk_assessments,
        messages_by_agent=messages_by_agent,
    )


def build_agent_messages(agent_name: str, prompt: str) -> list[dict[str, Any]]:
    """把 Agent Prompt 包装成聊天模型 messages。

    大模型接口通常不是直接接收一个字符串，
    而是接收 messages：

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]

    system:
        告诉模型它现在扮演什么角色，以及必须遵守的通用规则。

    user:
        放入这个 Agent 的完整任务 Prompt。
    """
    return [
        {
            "role": "system",
            "content": (
                f"你现在是 {agent_name}。"
                "请严格基于用户提供的数据和材料写报告。"
                "不要编造没有出现在材料里的具体数字、新闻或财务数据。"
                "本系统用于个人投资研究辅助，最终决策由使用者自行确认。"
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def call_agent_report(
    llm_client: ChatModelClient,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
) -> str:
    """调用大模型生成单个 Agent 报告。

    这个函数只做一件事：

        messages -> 大模型 -> assistant content

    如果模型返回格式不对，会抛出异常。
    这里暂时不做兜底，因为当前报告是自然语言报告，
    不是路由字段、交易动作这类必须结构化的关键控制信号。
    """
    response_json = llm_client.chat(
        messages=messages,
        temperature=temperature,
    )
    assistant_message = extract_assistant_message(response_json)
    report = assistant_message.get("content")

    if not isinstance(report, str) or not report.strip():
        raise ValueError("模型没有返回有效的报告文本。")

    messages.append(assistant_message)
    return report
