"""A 股无人值守自动模拟交易守护进程。

这个模块把系统从“用户问一只股票”升级成：

    程序自己定时扫描
      -> 发现候选股票
      -> 调用完整 TradingAgents 主链路
      -> 根据最终决策和风控护栏写入模拟盘
      -> 保存报告、full_state 和自动交易日志

重要边界：
    这里仍然只做模拟盘 paper trading。
    不连接券商，不真实下单，不自动花真钱交易。

为什么不让大模型 24 小时扫全市场？
    因为那样很慢、很贵，也容易把噪声当机会。
    正确做法是：
        先用行情数据和规则发现候选；
        再把少量候选交给 Agent 深度分析。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradingagents_cn.dataflows.stock_directory import find_stock_name_by_symbol
from tradingagents_cn.dataflows.stock_screening import (
    StockScreeningConfig,
    get_stock_screening_candidates,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol
from tradingagents_cn.graph.final_report import save_final_markdown_report
from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.graph.research_report_state_graph import run_research_report_state_graph
from tradingagents_cn.graph.run_state_logging import sanitize_path_part
from tradingagents_cn.graph.user_question_pipeline import build_report_path
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.paper_trading import load_paper_account
from tradingagents_cn.trading_rules import evaluate_a_share_buy_universe


AnalysisRunner = Callable[..., Any]
ScreeningLoader = Callable[[StockScreeningConfig], pd.DataFrame]


@dataclass(frozen=True)
class AutoTraderCandidate:
    """自动交易候选股票。

    source:
        候选来源：
            watchlist       自选股文件；
            holding         模拟账户已有持仓；
            market_screen   全市场行情快照筛选。

    trigger_reason:
        为什么这只股票进入本轮深度分析。
        这不是买入理由，只是“值得让 Agent 看一眼”的触发原因。
    """

    symbol: str
    name: str
    source: str
    trigger_reason: str
    score: float = 0.0
    latest_price: float | None = None
    change_pct: float | None = None
    amount: float | None = None
    turnover_rate: float | None = None
    volume_ratio: float | None = None


@dataclass
class AutoTraderRunItem:
    """一只候选股的自动分析结果摘要。"""

    symbol: str
    name: str
    source: str
    trigger_reason: str
    status: str
    report_path: str | None = None
    full_state_log_path: str | None = None
    trade_signal: str | None = None
    portfolio_rating: str | None = None
    risk_guardrail: str | None = None
    paper_trading_status: str | None = None
    paper_order_action: str | None = None
    paper_order_shares: int | None = None
    error: str = ""


@dataclass
class AutoTraderCycleResult:
    """一次自动扫描周期的结果。"""

    started_at: str
    finished_at: str
    trade_date: str
    market_phase: str
    paper_trading_enabled_this_cycle: bool
    candidates_found: int
    candidates_selected: int
    items: list[AutoTraderRunItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    log_path: str | None = None


@dataclass
class AutoTraderConfig:
    """自动模拟交易配置。

    run_forever:
        True 时进入长期循环；
        False 时只跑一轮，适合调试和测试。

    scan_interval_seconds:
        长期循环时，两次扫描之间休息多少秒。

    watchlist_path:
        自选股文件路径。
        每行可以写：
            000725
            000725,京东方A
        空行和 # 开头的注释会被忽略。

    include_market_screening:
        是否启用全市场行情快照筛选。

    include_holdings:
        是否总是复查模拟账户已有持仓。

    execute_only_during_trading_hours:
        True 时，非 A 股交易时段只分析不模拟下单。
        False 时，非交易时段也允许按最近价格做模拟成交。

    触发阈值：
        min_change_pct / max_drop_pct / min_turnover_rate / min_volume_ratio
        只用于决定“要不要深度分析”，不是买入标准。

    allowed_buy_universe:
        自动买入允许范围。
        当前固定为 main_board_common，也就是只允许沪深主板普通 A 股进入买入候选。
        已有持仓不会被这个字段过滤，因为已有仓位仍然需要复查和卖出。
    """

    run_forever: bool = False
    scan_interval_seconds: int = 300
    max_cycles: int | None = None
    watchlist_path: str | Path | None = None
    include_market_screening: bool = True
    include_holdings: bool = True
    max_candidates_per_cycle: int = 3
    candidate_pool_size: int = 30
    output_dir: str | Path = "outputs/auto_paper_trader"
    report_output_dir: str | Path = "outputs/user_questions"
    save_reports: bool = True
    execute_only_during_trading_hours: bool = True
    min_change_pct: float = 3.0
    max_drop_pct: float = -5.0
    min_turnover_rate: float = 5.0
    min_volume_ratio: float = 1.5
    allowed_buy_universe: str = "main_board_common"


class AutoPaperTrader:
    """无人值守自动模拟交易主控类。"""

    def __init__(
        self,
        auto_config: AutoTraderConfig | None = None,
        research_config: ResearchInputConfig | None = None,
        llm_client: Any | None = None,
        analysis_runner: AnalysisRunner = run_research_report_state_graph,
        screening_loader: ScreeningLoader = get_stock_screening_candidates,
        temperature: float = 0.2,
        max_debate_rounds: int = 1,
        max_risk_discuss_rounds: int = 1,
    ) -> None:
        """初始化自动交易器。"""
        self.auto_config = auto_config or AutoTraderConfig()
        self.research_config = research_config or ResearchInputConfig(
            enable_paper_trading=True
        )
        self.llm_client = llm_client
        self.analysis_runner = analysis_runner
        self.screening_loader = screening_loader
        self.temperature = temperature
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def run_forever(self) -> None:
        """长期运行自动扫描循环。

        这个函数会阻塞当前终端。
        日常调试先用 --once，确认没问题后再用 --loop。
        """
        cycle_count = 0
        while True:
            cycle_count += 1
            result = self.run_once()
            print(render_cycle_console_summary(result))

            if not self.auto_config.run_forever:
                return
            if (
                self.auto_config.max_cycles is not None
                and cycle_count >= self.auto_config.max_cycles
            ):
                return

            time.sleep(max(1, int(self.auto_config.scan_interval_seconds)))

    def run_once(self) -> AutoTraderCycleResult:
        """运行一轮自动扫描、分析和模拟交易。"""
        started = datetime.now()
        trade_date = date.today().strftime("%Y-%m-%d")
        market_phase = detect_market_phase(started)
        paper_enabled = self.should_enable_paper_trading(started)
        candidates, discovery_errors = self.discover_candidates()
        selected = candidates[: max(0, int(self.auto_config.max_candidates_per_cycle))]

        items: list[AutoTraderRunItem] = []
        for candidate in selected:
            items.append(
                self.run_candidate_analysis(
                    candidate=candidate,
                    trade_date=trade_date,
                    paper_enabled=paper_enabled,
                )
            )

        result = AutoTraderCycleResult(
            started_at=started.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            trade_date=trade_date,
            market_phase=market_phase,
            paper_trading_enabled_this_cycle=paper_enabled,
            candidates_found=len(candidates),
            candidates_selected=len(selected),
            items=items,
            errors=discovery_errors,
        )
        result.log_path = str(save_cycle_result(result, self.auto_config.output_dir))
        return result

    def discover_candidates(self) -> tuple[list[AutoTraderCandidate], list[str]]:
        """发现本轮需要深度分析的候选股票。"""
        candidates: list[AutoTraderCandidate] = []
        errors: list[str] = []

        if self.auto_config.watchlist_path:
            try:
                candidates.extend(
                    load_watchlist_candidates(
                        self.auto_config.watchlist_path,
                        allowed_buy_universe=self.auto_config.allowed_buy_universe,
                    )
                )
            except Exception as error:
                errors.append(f"读取自选股文件失败：{error}")

        if self.auto_config.include_holdings:
            try:
                candidates.extend(load_holding_candidates(self.research_config))
            except Exception as error:
                errors.append(f"读取模拟盘持仓失败：{error}")

        if self.auto_config.include_market_screening:
            try:
                candidates.extend(
                    discover_screening_candidates(
                        auto_config=self.auto_config,
                        screening_loader=self.screening_loader,
                    )
                )
            except Exception as error:
                errors.append(f"全市场候选筛选失败：{error}")

        return deduplicate_candidates(candidates), errors

    def run_candidate_analysis(
        self,
        candidate: AutoTraderCandidate,
        trade_date: str,
        paper_enabled: bool,
    ) -> AutoTraderRunItem:
        """对一只候选股票运行完整 TradingAgents 链路。"""
        try:
            config_for_cycle = replace(
                self.research_config,
                enable_paper_trading=paper_enabled,
            )
            result = self.analysis_runner(
                symbol=candidate.symbol,
                trade_date=trade_date,
                config=config_for_cycle,
                llm_client=self.get_llm_client(),
                temperature=self.temperature,
                max_debate_rounds=self.max_debate_rounds,
                max_risk_discuss_rounds=self.max_risk_discuss_rounds,
                thread_id=f"auto-paper-{candidate.symbol}-{trade_date}",
            )

            report_path = None
            if self.auto_config.save_reports:
                report_path = build_report_path(
                    output_dir=self.auto_config.report_output_dir,
                    symbol=candidate.symbol,
                    stock_name=candidate.name,
                    trade_date=trade_date,
                )
                save_final_markdown_report(result, report_path)

            return build_run_item_from_result(
                candidate=candidate,
                result=result,
                report_path=report_path,
            )
        except Exception as error:
            return AutoTraderRunItem(
                symbol=candidate.symbol,
                name=candidate.name,
                source=candidate.source,
                trigger_reason=candidate.trigger_reason,
                status="error",
                error=str(error),
            )

    def get_llm_client(self) -> Any:
        """获取或创建模型客户端。"""
        if self.llm_client is None:
            self.llm_client = create_chat_client()
        return self.llm_client

    def should_enable_paper_trading(self, now: datetime) -> bool:
        """判断本轮是否允许写入模拟盘成交。"""
        if not self.research_config.enable_paper_trading:
            return False
        if not self.auto_config.execute_only_during_trading_hours:
            return True
        return is_a_share_trading_time(now)


def load_watchlist_candidates(
    path: str | Path,
    allowed_buy_universe: str = "main_board_common",
) -> list[AutoTraderCandidate]:
    """从自选股文件读取候选股票。"""
    watchlist_path = Path(path).expanduser()
    if not watchlist_path.exists():
        return []

    candidates: list[AutoTraderCandidate] = []
    for line in watchlist_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue

        parts = [part.strip() for part in text.replace("，", ",").split(",")]
        symbol = normalize_cn_symbol(parts[0])
        name = parts[1] if len(parts) > 1 and parts[1] else find_stock_name_by_symbol(symbol)
        buy_rule = evaluate_a_share_buy_universe(
            symbol=symbol,
            name=name,
            allowed_universe=allowed_buy_universe,
        )
        if not buy_rule.allowed_to_buy:
            continue

        candidates.append(
            AutoTraderCandidate(
                symbol=symbol,
                name=name or symbol,
                source="watchlist",
                trigger_reason="自选股定时复查。",
                score=100.0,
            )
        )
    return candidates


def load_holding_candidates(config: ResearchInputConfig) -> list[AutoTraderCandidate]:
    """从模拟账户读取已有持仓，作为必须复查的候选。"""
    account = load_paper_account(
        config.paper_trading_ledger_path,
        initial_cash=config.paper_trading_initial_cash,
    )
    candidates: list[AutoTraderCandidate] = []
    for symbol, position in account.positions.items():
        if position.shares <= 0:
            continue
        candidates.append(
            AutoTraderCandidate(
                symbol=symbol,
                name=find_stock_name_by_symbol(symbol) or symbol,
                source="holding",
                trigger_reason=f"模拟账户已有持仓 {position.shares} 股，需要定时复查风险。",
                score=90.0,
                latest_price=position.last_price or None,
            )
        )
    return candidates


def discover_screening_candidates(
    auto_config: AutoTraderConfig,
    screening_loader: ScreeningLoader = get_stock_screening_candidates,
) -> list[AutoTraderCandidate]:
    """从全市场行情快照里发现候选股票。"""
    frame = screening_loader(
        StockScreeningConfig(max_candidates=auto_config.candidate_pool_size)
    )
    if frame is None or frame.empty:
        return []

    candidates: list[AutoTraderCandidate] = []
    for _, row in frame.iterrows():
        candidate = build_candidate_from_screening_row(row, auto_config)
        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def build_candidate_from_screening_row(
    row: Any,
    auto_config: AutoTraderConfig,
) -> AutoTraderCandidate | None:
    """把行情快照一行转换成候选股票。"""
    symbol = normalize_cn_symbol(str(row.get("Symbol") or ""))
    if not symbol:
        return None

    name = str(row.get("Name") or find_stock_name_by_symbol(symbol) or symbol)
    buy_rule = evaluate_a_share_buy_universe(
        symbol=symbol,
        name=name,
        allowed_universe=auto_config.allowed_buy_universe,
    )
    if not buy_rule.allowed_to_buy:
        return None

    latest_price = to_optional_float(row.get("Latest"))
    change_pct = to_optional_float(row.get("ChangePct"))
    amount = to_optional_float(row.get("Amount"))
    turnover_rate = to_optional_float(row.get("TurnoverRate"))
    volume_ratio = to_optional_float(row.get("VolumeRatio"))

    reasons: list[str] = []
    score = 0.0

    if change_pct is not None and change_pct >= auto_config.min_change_pct:
        reasons.append(f"涨幅 {change_pct:.2f}% 达到强势触发线。")
        score += change_pct

    if change_pct is not None and change_pct <= auto_config.max_drop_pct:
        reasons.append(f"跌幅 {change_pct:.2f}% 达到风险复查触发线。")
        score += abs(change_pct)

    if turnover_rate is not None and turnover_rate >= auto_config.min_turnover_rate:
        reasons.append(f"换手率 {turnover_rate:.2f}% 较高。")
        score += turnover_rate * 0.5

    if volume_ratio is not None and volume_ratio >= auto_config.min_volume_ratio:
        reasons.append(f"量比 {volume_ratio:.2f} 放大。")
        score += volume_ratio * 2

    if not reasons:
        return None

    return AutoTraderCandidate(
        symbol=symbol,
        name=name,
        source="market_screen",
        trigger_reason="；".join(reasons),
        score=round(score, 4),
        latest_price=latest_price,
        change_pct=change_pct,
        amount=amount,
        turnover_rate=turnover_rate,
        volume_ratio=volume_ratio,
    )


def deduplicate_candidates(
    candidates: list[AutoTraderCandidate],
) -> list[AutoTraderCandidate]:
    """按股票代码去重，并保留更高优先级候选。"""
    priority = {
        "holding": 3,
        "watchlist": 2,
        "market_screen": 1,
    }
    best: dict[str, AutoTraderCandidate] = {}
    for candidate in candidates:
        symbol = normalize_cn_symbol(candidate.symbol)
        existing = best.get(symbol)
        if existing is None:
            best[symbol] = candidate
            continue

        current_key = (priority.get(candidate.source, 0), candidate.score)
        existing_key = (priority.get(existing.source, 0), existing.score)
        if current_key > existing_key:
            best[symbol] = candidate

    return sorted(
        best.values(),
        key=lambda item: (priority.get(item.source, 0), item.score),
        reverse=True,
    )


def build_run_item_from_result(
    candidate: AutoTraderCandidate,
    result: Any,
    report_path: Path | None,
) -> AutoTraderRunItem:
    """把完整研究结果压缩成自动交易日志摘要。"""
    paper = getattr(result, "paper_trading_result", {}) or {}
    order = paper.get("order") or {}
    return AutoTraderRunItem(
        symbol=candidate.symbol,
        name=candidate.name,
        source=candidate.source,
        trigger_reason=candidate.trigger_reason,
        status="ok",
        report_path=str(report_path) if report_path is not None else None,
        full_state_log_path=str(getattr(result, "full_state_log_path", "") or ""),
        trade_signal=str(getattr(getattr(result, "trade_signal", None), "action", "")),
        portfolio_rating=extract_enum_value(
            getattr(getattr(result, "portfolio_decision", None), "rating", "")
        ),
        risk_guardrail=str(
            getattr(getattr(result, "risk_guardrail", None), "chinese_summary", "")
        ),
        paper_trading_status=str(paper.get("status", "")),
        paper_order_action=str(order.get("action", "")) if order else None,
        paper_order_shares=to_optional_int(order.get("shares")) if order else None,
    )


def save_cycle_result(
    result: AutoTraderCycleResult,
    output_dir: str | Path,
) -> Path:
    """保存一次自动交易扫描周期日志。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_time = sanitize_path_part(result.started_at.replace(" ", "_").replace(":", "-"))
    path = root / f"{safe_time}_cycle.json"
    payload = asdict(result)
    payload["log_path"] = str(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def render_cycle_console_summary(result: AutoTraderCycleResult) -> str:
    """渲染给终端看的自动交易周期摘要。"""
    lines = [
        "========== 自动模拟交易周期完成 ==========",
        f"时间：{result.started_at} -> {result.finished_at}",
        f"交易日：{result.trade_date}",
        f"市场阶段：{result.market_phase}",
        f"本轮是否允许模拟成交：{'是' if result.paper_trading_enabled_this_cycle else '否'}",
        f"候选发现/实际分析：{result.candidates_found}/{result.candidates_selected}",
        f"日志：{result.log_path}",
    ]
    if result.errors:
        lines.append("发现阶段错误：")
        lines.extend(f"- {error}" for error in result.errors)

    if result.items:
        lines.append("分析结果：")
        for item in result.items:
            lines.append(
                f"- {item.name}（{item.symbol}）：{item.trade_signal or '未知信号'}，"
                f"模拟盘 {item.paper_trading_status or '未启用'}，{item.trigger_reason}"
            )
    return "\n".join(lines)


def detect_market_phase(now: datetime) -> str:
    """判断当前处于 A 股哪个时间段。"""
    if now.weekday() >= 5:
        return "weekend"
    current = now.time()
    if dt_time(9, 25) <= current <= dt_time(11, 30):
        return "morning_trading"
    if dt_time(13, 0) <= current <= dt_time(15, 0):
        return "afternoon_trading"
    if dt_time(8, 0) <= current < dt_time(9, 25):
        return "pre_market"
    if dt_time(11, 30) < current < dt_time(13, 0):
        return "midday_break"
    if dt_time(15, 0) < current <= dt_time(18, 0):
        return "after_market"
    return "overnight"


def is_a_share_trading_time(now: datetime) -> bool:
    """判断当前是否是 A 股连续竞价交易时段。"""
    return detect_market_phase(now) in {"morning_trading", "afternoon_trading"}


def to_optional_float(value: Any) -> float | None:
    """安全转 float。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_optional_int(value: Any) -> int | None:
    """安全转 int。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_enum_value(value: Any) -> str:
    """提取 Enum/Pydantic 字段的 value。"""
    return str(getattr(value, "value", value) or "")
