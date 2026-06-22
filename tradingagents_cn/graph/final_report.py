"""最终 Markdown 报告渲染。

这个文件只做一件事：

    把 run_research_report_pipeline(...) 返回的 result
    渲染成一份完整 Markdown 报告。

它不负责：

    - 获取数据；
    - 调用大模型；
    - 做 Agent 路由；
    - 修改任何交易决策。

为什么要单独拆出来？
    因为报告展示属于“输出层”。
    后续你可能会有多种输出：
        1. Markdown 文件；
        2. 控制台文本；
        3. Web 页面；
        4. Excel / Word / PDF。

    如果把排版代码塞进主流程，后续会很乱。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tradingagents_cn.dataflows.stock_directory import find_stock_name_by_symbol
from tradingagents_cn.graph.display_text import (
    localize_report_text,
    render_selected_analysts_cn,
    translate_machine_action,
    translate_paper_status,
    translate_portfolio_rating,
    translate_trader_action,
)
from tradingagents_cn.graph.research_report_pipeline import ResearchReportPipelineResult
from tradingagents_cn.graph.signal_processing import render_risk_guardrail_decision


@dataclass(frozen=True)
class TradingStatus:
    """给人看的交易状态。

    结构化 Agent 内部仍然使用：
        Buy / Overweight / Hold / Underweight / Sell

    但用户真正关心的是：
        现在能不能买；
        如果不能买，是永远回避，还是等回调；
        如果已经持有，该怎么处理。
    """

    code: str
    label: str
    summary: str
    current_action: str
    if_holding: str
    if_not_holding: str
    watch_condition: str


def render_final_markdown_report(result: ResearchReportPipelineResult) -> str:
    """把完整研究结果渲染成 Markdown。

    参数：
        result:
            run_research_report_pipeline(...) 的返回结果。

    返回：
        Markdown 字符串。
    """
    final_state = result.final_state
    symbol = final_state.get("symbol", "未知股票")
    stock_name = find_stock_name_by_symbol(symbol) or symbol
    trade_date = final_state.get("trade_date", "未知日期")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# A 股多智能体研究报告：{stock_name}（{symbol}）",
        "",
        "## 最终结论",
        "",
        render_user_facing_conclusion(result),
        "",
        "## 报告摘要",
        "",
        f"- 股票名称：{stock_name}",
        f"- 股票代码：{symbol}",
        f"- 分析日期：{trade_date}",
        f"- 报告生成时间：{generated_at}",
        f"- 多空辩论轮数：{result.max_debate_rounds}",
        f"- 启用分析员：{render_selected_analysts(result.selected_analysts)}",
        f"- 研究经理评级：{translate_portfolio_rating(result.research_plan.recommendation.value)}",
        f"- 交易员动作：{translate_trader_action(result.trader_proposal.action.value)}",
        f"- 组合经理最终评级：{translate_portfolio_rating(result.portfolio_decision.rating.value)}",
        f"- 交易状态：{build_trading_status(result).label}",
        f"- 机器交易信号：{translate_machine_action(result.trade_signal.action)}，{result.trade_signal.chinese_action}",
        f"- 风控护栏：{result.risk_guardrail.chinese_summary}",
        f"- 模拟盘：{render_paper_trading_brief(result)}",
        f"- 完整状态日志：{result.full_state_log_path or '未保存'}",
        "",
        "本报告用于个人投资研究辅助，最终决策由使用者自行确认。",
        "",
        "## 数据获取问题",
        "",
        render_data_errors(result.data_errors),
        "",
        "## 组合经理最终交易决策",
        "",
        localize_report_text(result.final_trade_decision),
        "",
        "## 程序风控护栏",
        "",
        localize_report_text(render_risk_guardrail_decision(result.risk_guardrail)),
        "",
        "## 模拟盘自动交易",
        "",
        render_paper_trading_result(result),
        "",
        "## 交易员交易提案",
        "",
        localize_report_text(result.trader_plan),
        "",
        "## 研究经理研究计划",
        "",
        localize_report_text(result.investment_plan),
        "",
        "## 风险控制辩论",
        "",
        localize_report_text(result.risk_debate_history),
        "",
        "## 多头和空头辩论",
        "",
        localize_report_text(result.debate_history),
        "",
        "## 综合研究结论",
        "",
        localize_report_text(result.summary_report),
        "",
        "## 技术面报告",
        "",
        localize_report_text(result.market_report),
        "",
        "## 情绪面报告",
        "",
        localize_report_text(result.sentiment_report),
        "",
        "## 新闻公告报告",
        "",
        localize_report_text(result.news_report),
        "",
        "## 基本面报告",
        "",
        localize_report_text(result.fundamentals_report),
        "",
        "## 调试信息",
        "",
        render_debug_summary(result),
        "",
    ]

    return "\n".join(sections)


def save_final_markdown_report(
    result: ResearchReportPipelineResult,
    output_path: str | Path,
) -> Path:
    """把最终 Markdown 报告保存到文件。

    参数：
        result:
            run_research_report_pipeline(...) 的返回结果。

        output_path:
            要保存的文件路径。

    返回：
        保存后的 Path。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_final_markdown_report(result), encoding="utf-8")
    return path


def render_data_errors(data_errors: list[str]) -> str:
    """渲染数据获取错误。"""
    if not data_errors:
        return "无。"

    return "\n".join(f"- {error}" for error in data_errors)


def render_user_facing_conclusion(result: ResearchReportPipelineResult) -> str:
    """渲染给人看的最终操作结论。

    前面的 Agent 内部仍然保留英文枚举值：
        Buy / Overweight / Hold / Underweight / Sell

    但最终报告是给人读的，
    所以这里把英文枚举翻译成更直白的中文：
        到底是偏买、偏卖、观望，还是降低仓位。
    """
    trader_action = result.trader_proposal.action.value
    portfolio_rating = result.portfolio_decision.rating.value
    research_rating = result.research_plan.recommendation.value

    status = build_trading_status(result)
    position_sizing = result.trader_proposal.position_sizing or "模型没有给出明确仓位比例。"
    executive_summary = result.portfolio_decision.executive_summary

    lines = [
        f"**操作结论：{status.summary}**",
        "",
        f"- 交易状态：{status.label}",
        f"- 现在能不能买：{status.current_action}",
        f"- 如果已经持有：{status.if_holding}",
        f"- 如果还没持有：{status.if_not_holding}",
        f"- 等待条件：{status.watch_condition}",
        f"- 交易员动作：{translate_trader_action(trader_action)}",
        f"- 组合经理最终评级：{translate_portfolio_rating(portfolio_rating)}",
        f"- 研究经理评级：{translate_portfolio_rating(research_rating)}",
        f"- 机器交易信号：{translate_machine_action(result.trade_signal.action)}，{result.trade_signal.chinese_action}",
        f"- 仓位建议：{position_sizing}",
        f"- 程序风控护栏：{result.risk_guardrail.chinese_summary}",
        f"- 模拟盘：{render_paper_trading_brief(result)}",
        f"- 核心理由：{executive_summary}",
    ]
    return "\n".join(lines)


def build_trading_status(result: ResearchReportPipelineResult) -> TradingStatus:
    """把机器评级转换成更适合人使用的交易状态。

    这个函数不负责预测涨跌，也不覆盖 Agent 的结构化评级。
    它只是把“卖出 / 低配 / 持有”进一步解释成更细的人话状态：
        - 立即买入；
        - 小仓位试探；
        - 等待回调；
        - 加入观察池；
        - 持有观察；
        - 减仓风控；
        - 卖出回避。
    """
    trader_action = result.trader_proposal.action.value
    portfolio_rating = result.portfolio_decision.rating.value
    risk_guardrail = result.risk_guardrail
    evidence_text = collect_status_evidence_text(result)

    has_positive_quality = contains_any(
        evidence_text,
        [
            "基本面强劲",
            "基本面较强",
            "基本面韧性",
            "盈利改善",
            "盈利能力改善",
            "业绩高增长",
            "高增长",
            "行业龙头",
            "现金流强劲",
            "订单增长",
            "需求旺盛",
            "景气度",
            "净利润同比增长",
            "营收增长",
            "毛利率改善",
        ],
    )
    has_overheat_or_pullback = contains_any(
        evidence_text,
        [
            "超买",
            "布林带上轨",
            "突破布林带",
            "涨超",
            "短期涨幅",
            "追高",
            "回调",
            "乖离",
            "高位",
            "量价背离",
            "波动风险",
            "风险收益比不佳",
        ],
    )
    has_watch_condition = contains_any(
        evidence_text,
        [
            "等待",
            "确认",
            "半年报",
            "回调至",
            "回调到",
            "企稳",
            "支撑",
            "中轨",
            "均线",
            "重新评估",
            "再评估",
        ],
    )
    has_severe_avoid_risk = contains_any(
        evidence_text,
        [
            "全年亏损",
            "商誉减值",
            "监管风险",
            "业务占比仅",
            "无收入",
            "退市",
            "立案",
            "处罚",
            "基本面与股价严重脱节",
            "现金流恶化",
            "净利润同比下滑",
        ],
    )

    allow_new_position = bool(getattr(risk_guardrail, "allow_new_position", False))
    risk_band = str(getattr(risk_guardrail, "risk_band", "blocked") or "blocked")

    if portfolio_rating in {"Buy", "Overweight"} and allow_new_position:
        if risk_band in {"controlled", "defensive"}:
            return TradingStatus(
                code="SMALL_PROBE",
                label="小仓位试探",
                summary="可以考虑小仓位试探，但必须严格按风控仓位和止损条件执行。",
                current_action="可以小仓位参与，不适合一次性重仓。",
                if_holding="已有仓位可以继续持有，但不宜突破风控上限。",
                if_not_holding="未持有者可以分批小仓位试探。",
                watch_condition="价格、成交量和风险触发条件继续满足报告要求。",
            )
        return TradingStatus(
            code="BUY_NOW",
            label="可以考虑买入",
            summary="可以考虑买入或加仓，但仍要按仓位建议分批执行。",
            current_action="可以买，但要分批，不要一次性满仓。",
            if_holding="已有仓位可以按计划加仓，但不要超过仓位上限。",
            if_not_holding="未持有者可以按计划建立底仓。",
            watch_condition="入场价、止损价和仓位上限都按报告执行。",
        )

    if portfolio_rating == "Hold" or trader_action == "Hold":
        return TradingStatus(
            code="HOLD_OBSERVE",
            label="持有观察",
            summary="暂时观望，不建议因为单次报告直接重仓买入或卖出。",
            current_action="现在不主动买入，先观察。",
            if_holding="已有仓位可以继续观察，按报告设置止损或复评条件。",
            if_not_holding="未持有者暂不追买，等待更明确的买点。",
            watch_condition="等待趋势、成交量、基本面或消息面出现更明确确认。",
        )

    if has_positive_quality and has_overheat_or_pullback:
        return TradingStatus(
            code="WAIT_PULLBACK",
            label="等待回调",
            summary="不建议追高买入，但这类股票可以加入观察池，等待回调后的买点。",
            current_action="现在不追高买入。",
            if_holding="已有仓位可先降到舒服仓位，保留观察仓，避免高位扩大风险。",
            if_not_holding="未持有者先放入观察池，等回调或企稳信号。",
            watch_condition="等待回调到报告提到的支撑位、均线、布林带中轨，或等待成交缩量企稳。",
        )

    if has_severe_avoid_risk and portfolio_rating in {"Sell", "Underweight"}:
        return TradingStatus(
            code="SELL_AVOID",
            label="卖出回避",
            summary="不建议买入；如果已经持有，建议考虑减仓或卖出。",
            current_action="现在不买，并且不作为优先观察对象。",
            if_holding="已有仓位按报告考虑减仓或退出。",
            if_not_holding="未持有者回避，除非后续基本面或风险项发生明显改变。",
            watch_condition="等待重大风险解除、财务质量修复或监管/公告不确定性消除。",
        )

    if has_watch_condition:
        return TradingStatus(
            code="WATCHLIST",
            label="加入观察池",
            summary="当前不建议买入，但可以放入观察池，等待条件确认后再评估。",
            current_action="现在不买，先等条件。",
            if_holding="已有仓位可降低风险敞口，保留少量观察仓。",
            if_not_holding="未持有者先不建仓，等待报告里的触发条件。",
            watch_condition="等待报告提到的财报、现金流、价格支撑、趋势企稳或消息确认。",
        )

    if portfolio_rating in {"Underweight", "Sell"} or trader_action == "Sell":
        return TradingStatus(
            code="REDUCE_RISK",
            label="减仓风控",
            summary="偏谨慎，不建议追买；已有仓位建议降低风险敞口。",
            current_action="现在不新增仓位。",
            if_holding="已有仓位考虑减仓，把风险降到可承受范围。",
            if_not_holding="未持有者等待更好的风险收益比。",
            watch_condition="等待风险释放、价格回调和基本面验证。",
        )

    return TradingStatus(
        code="WATCHLIST",
        label="加入观察池",
        summary="当前信号不够明确，建议先加入观察池，等待更多确认。",
        current_action="现在不急着买。",
        if_holding="已有仓位按原计划控制风险。",
        if_not_holding="未持有者先观察，不追高。",
        watch_condition="等待下一次数据、行情和报告共同确认。",
    )


def collect_status_evidence_text(result: ResearchReportPipelineResult) -> str:
    """收集交易状态判断需要看的报告文本。"""
    portfolio_decision = result.portfolio_decision
    parts = [
        getattr(portfolio_decision, "executive_summary", ""),
        getattr(portfolio_decision, "investment_thesis", ""),
        getattr(result, "investment_plan", ""),
        getattr(result, "trader_plan", ""),
        getattr(result, "summary_report", ""),
        getattr(result, "market_report", ""),
        getattr(result, "news_report", ""),
        getattr(result, "fundamentals_report", ""),
    ]
    return "\n".join(str(part or "") for part in parts)


def contains_any(text: str, keywords: list[str]) -> bool:
    """判断文本里是否出现任意关键词。"""
    return any(keyword in text for keyword in keywords)


def render_paper_trading_brief(result: ResearchReportPipelineResult) -> str:
    """渲染一句话模拟盘结果。"""
    paper = result.paper_trading_result or {}
    if not paper or paper.get("enabled") is False:
        return "未启用。"

    status = str(paper.get("status") or "unknown")
    order = paper.get("order") or {}
    if status == "filled":
        return (
            f"已模拟成交 {translate_machine_action(str(order.get('action') or ''))} "
            f"{order.get('shares')} 股，成交价 {order.get('price')}。"
        )
    if status == "skipped":
        return f"已启用但未成交，原因：{order.get('reason', '未记录原因')}"
    if status == "price_unavailable":
        return f"已启用但无法获取模拟成交价：{paper.get('error', '未知错误')}"
    if status == "error":
        return f"执行异常：{paper.get('error', '未知错误')}"
    return f"已启用，状态：{translate_paper_status(status)}"


def render_paper_trading_result(result: ResearchReportPipelineResult) -> str:
    """渲染模拟盘自动交易详情。"""
    paper = result.paper_trading_result or {}
    if not paper or paper.get("enabled") is False:
        return "未启用模拟盘。运行 CLI 时添加 `--paper-trading` 才会写入本地模拟账户。"

    lines = [
        f"- 状态：{translate_paper_status(str(paper.get('status') or 'unknown'))}",
        f"- 账户文件：{paper.get('ledger_path', '未记录')}",
        f"- 本轮复盘待处理成交数：{paper.get('reviewed_pending_count', 0)}",
    ]
    if "execution_price" in paper:
        lines.append(f"- 模拟成交参考价：{paper['execution_price']}")

    order = paper.get("order") or {}
    if order:
        lines.extend(
            [
                f"- 模拟动作：{translate_machine_action(str(order.get('action') or ''))}",
                f"- 成交状态：{translate_paper_status(str(order.get('status') or ''))}",
                f"- 股数：{order.get('shares')}",
                f"- 金额：{order.get('amount')}",
                f"- 原因：{order.get('reason')}",
            ]
        )

    account = paper.get("account") or {}
    if account:
        lines.extend(
            [
                f"- 账户现金：{account.get('cash')}",
                f"- 账户总资产估算：{account.get('total_equity')}",
                f"- 累计模拟收益率：{format_percent_value(account.get('total_return'))}",
                f"- 当前持仓数量：{account.get('position_count')}",
                f"- 历史模拟记录数：{account.get('trade_count')}",
            ]
        )
    if paper.get("error"):
        lines.append(f"- 错误：{paper['error']}")
    return "\n".join(lines)


def format_percent_value(value: object) -> str:
    """把小数格式化成百分比。"""
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "未记录"


def build_operation_sentence(trader_action: str, portfolio_rating: str) -> str:
    """根据最终交易动作和组合评级，生成一句明确的人话结论。

    判断优先级：
        1. Trader 的 Buy / Hold / Sell 是最直接的买卖动作；
        2. Portfolio 的 Buy / Overweight / Hold / Underweight / Sell
           用来补充仓位态度；
        3. 如果两者出现轻微差异，以更保守的说法表达。
    """
    if trader_action == "Buy":
        if portfolio_rating in {"Buy", "Overweight"}:
            return "可以考虑买入或加仓，但仍要按仓位建议分批执行。"
        return "可以关注买入机会，但组合评级不够强，建议小仓位试探。"

    if trader_action == "Sell":
        if portfolio_rating in {"Sell", "Underweight"}:
            return "不建议买入；如果已经持有，建议考虑减仓或卖出。"
        return "短线不建议买入；如果已经持有，建议先降低仓位观察。"

    if portfolio_rating == "Buy":
        return "偏积极，可以考虑买入，但交易员暂未给出强买入动作。"

    if portfolio_rating == "Overweight":
        return "偏积极，可以考虑低吸或分批加仓，但不适合一次性重仓。"

    if portfolio_rating == "Underweight":
        return "偏谨慎，不建议追买；已有仓位建议降低风险敞口。"

    if portfolio_rating == "Sell":
        return "偏空，不建议买入；已有仓位建议考虑退出。"

    return "暂时观望，不建议因为单次报告直接重仓买入或卖出。"


def render_debug_summary(result: ResearchReportPipelineResult) -> str:
    """渲染调试摘要。

    这里只列出各 Agent 的消息 key，
    不展开完整 Prompt，
    避免最终报告过长。
    """
    message_keys = sorted(result.messages_by_agent.keys())
    lines = [
        "### 分析员消息键",
        "",
    ]
    lines.extend(f"- {key}" for key in message_keys)
    if result.full_state_log_path is not None:
        lines.extend(
            [
                "",
                f"full_state.json：{result.full_state_log_path}",
            ]
        )
    return "\n".join(lines)


def render_selected_analysts(selected_analysts: tuple[str, ...]) -> str:
    """把 selected_analysts 渲染成人可读文本。"""
    return render_selected_analysts_cn(selected_analysts)
