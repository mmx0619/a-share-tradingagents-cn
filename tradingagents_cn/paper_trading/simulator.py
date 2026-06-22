"""A 股模拟盘自动交易执行器。

这个模块只做“模拟盘 / paper trading”，不接任何券商接口。

它接在 Portfolio Manager 和程序风控护栏之后：

    Portfolio Manager 最终评级
      -> TradeSignal 机器信号
      -> RiskGuardrailDecision 风控护栏
      -> Paper Trading 模拟买卖
      -> 本地 account.json

为什么要单独拆一个模块？
    1. 投研报告是“分析系统”；
    2. 模拟盘是“执行和复盘系统”；
    3. 两者应该连接，但不要混成一坨。

注意：
    本模块不会真实下单。
    所有成交、现金、持仓都只写在本地 JSON 文件里。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from tradingagents_cn.dataflows.stock_directory import find_stock_name_by_symbol
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol
from tradingagents_cn.dataflows.vendor_router import route_daily_history, route_realtime_quote
from tradingagents_cn.memory.outcome import resolve_decision_outcome
from tradingagents_cn.trading_rules import evaluate_a_share_buy_universe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_LEDGER_PATH = PROJECT_ROOT / "outputs" / "paper_trading" / "account.json"


@dataclass
class PaperPosition:
    """模拟盘持仓。

    symbol:
        股票代码。

    shares:
        当前持股数量。

    avg_cost:
        持仓平均成本。

    last_price / market_value / unrealized_return:
        最近一次更新价格、持仓市值、浮动收益率。
        这些字段用于看账户状态，不用于真实交易。
    """

    symbol: str
    shares: int
    avg_cost: float
    last_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_return: float = 0.0
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperPosition":
        """从 JSON 字典恢复持仓对象。"""
        return cls(
            symbol=normalize_cn_symbol(str(data.get("symbol", ""))),
            shares=int(data.get("shares", 0) or 0),
            avg_cost=float(data.get("avg_cost", 0.0) or 0.0),
            last_price=float(data.get("last_price", 0.0) or 0.0),
            market_value=float(data.get("market_value", 0.0) or 0.0),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0) or 0.0),
            unrealized_return=float(data.get("unrealized_return", 0.0) or 0.0),
            updated_at=str(data.get("updated_at", "") or ""),
        )


@dataclass
class PaperTrade:
    """模拟成交记录。

    这里记录的是“程序实际模拟执行了什么”，不是长篇投研报告。

    status:
        filled 表示模拟成交成功。
        skipped 表示本次信号没有产生订单。

    review_status:
        pending 表示这笔模拟交易还没有做收益复盘。
        reviewed 表示已经根据后续行情复盘过。
        not_required 表示没有必要复盘，例如 skipped。
    """

    trade_id: str
    symbol: str
    trade_date: str
    action: str
    shares: int
    price: float
    amount: float
    cash_after: float
    position_after: int
    reason: str
    status: str
    source_rating: str
    source_signal: str
    created_at: str
    review_status: str = "pending"
    raw_return: float | None = None
    alpha_return: float | None = None
    holding_days: int | None = None
    reflection: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperTrade":
        """从 JSON 字典恢复成交记录。"""
        return cls(
            trade_id=str(data.get("trade_id", "")),
            symbol=normalize_cn_symbol(str(data.get("symbol", ""))),
            trade_date=str(data.get("trade_date", "")),
            action=str(data.get("action", "")),
            shares=int(data.get("shares", 0) or 0),
            price=float(data.get("price", 0.0) or 0.0),
            amount=float(data.get("amount", 0.0) or 0.0),
            cash_after=float(data.get("cash_after", 0.0) or 0.0),
            position_after=int(data.get("position_after", 0) or 0),
            reason=str(data.get("reason", "")),
            status=str(data.get("status", "")),
            source_rating=str(data.get("source_rating", "")),
            source_signal=str(data.get("source_signal", "")),
            created_at=str(data.get("created_at", "")),
            review_status=str(data.get("review_status", "pending") or "pending"),
            raw_return=optional_float(data.get("raw_return")),
            alpha_return=optional_float(data.get("alpha_return")),
            holding_days=optional_int(data.get("holding_days")),
            reflection=str(data.get("reflection", "") or ""),
        )


@dataclass
class PaperAccount:
    """模拟盘账户。

    cash:
        当前模拟现金。

    positions:
        当前持仓，key 是股票代码。

    trades:
        历史模拟成交。

    initial_cash:
        初始模拟资金。
    """

    cash: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    trades: list[PaperTrade] = field(default_factory=list)
    initial_cash: float = 10000.0
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], initial_cash: float) -> "PaperAccount":
        """从 JSON 字典恢复账户对象。"""
        positions = {
            normalize_cn_symbol(symbol): PaperPosition.from_dict(position)
            for symbol, position in (data.get("positions") or {}).items()
        }
        trades = [
            PaperTrade.from_dict(item)
            for item in (data.get("trades") or [])
            if isinstance(item, dict)
        ]
        return cls(
            cash=float(data.get("cash", initial_cash) or initial_cash),
            positions=positions,
            trades=trades,
            initial_cash=float(data.get("initial_cash", initial_cash) or initial_cash),
            updated_at=str(data.get("updated_at", "") or ""),
        )


@dataclass
class PaperTradingConfig:
    """模拟盘配置。

    enabled:
        是否启用模拟盘自动交易。
        默认关闭，避免用户只是生成报告时意外写账户。

    ledger_path:
        本地账户 JSON 文件路径。

    max_single_position_pct:
        单只股票最大模拟仓位。
        它是账户层全局上限，还会和 RiskGuardrailDecision 的上限取更小值。

    min_trade_amount:
        最小模拟成交金额。
        太小的订单会被跳过，避免账户里出现很多没意义的小单。

    lot_size:
        A 股买入最小单位通常是 100 股。

    buy_universe:
        自动买入范围。
        当前固定使用 main_board_common，也就是只允许买沪深主板普通 A 股。
        这个规则是程序硬门禁，大模型给出 BUY 也不能绕过。
    """

    enabled: bool = False
    ledger_path: str | Path = DEFAULT_PAPER_LEDGER_PATH
    initial_cash: float = 10000.0
    max_single_position_pct: float = 0.20
    min_trade_amount: float = 1000.0
    lot_size: int = 100
    review_pending: bool = True
    review_holding_days: int = 5
    benchmark_symbol: str = "000300"
    benchmark_name: str = "沪深300"
    buy_universe: str = "main_board_common"

    @classmethod
    def from_research_config(cls, config: Any) -> "PaperTradingConfig":
        """从 ResearchInputConfig 提取模拟盘配置。

        这里用 getattr，是为了避免 simulator 反向强依赖 graph 配置类。
        """
        return cls(
            enabled=bool(getattr(config, "enable_paper_trading", False)),
            ledger_path=getattr(
                config,
                "paper_trading_ledger_path",
                DEFAULT_PAPER_LEDGER_PATH,
            ),
            initial_cash=float(getattr(config, "paper_trading_initial_cash", 10000.0)),
            max_single_position_pct=float(
                getattr(config, "paper_trading_max_single_position_pct", 0.20)
            ),
            min_trade_amount=float(getattr(config, "paper_trading_min_trade_amount", 1000.0)),
            review_pending=bool(getattr(config, "paper_trading_review_pending", True)),
            review_holding_days=int(
                getattr(
                    config,
                    "paper_trading_review_holding_days",
                    getattr(config, "memory_holding_days", 5),
                )
            ),
            benchmark_symbol=str(getattr(config, "benchmark_symbol", "000300")),
            benchmark_name=str(getattr(config, "benchmark_name", "沪深300")),
            buy_universe=str(
                getattr(config, "paper_trading_buy_universe", "main_board_common")
            ),
        )


def optional_float(value: Any) -> float | None:
    """把可选字段转成 float。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    """把可选字段转成 int。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_ledger_path(path: str | Path) -> Path:
    """解析账户文件路径。

    用户如果传相对路径，就认为它相对于项目根目录。
    这样不管从哪里运行 CLI，默认账户文件都在项目 outputs 下。
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_paper_account(path: str | Path, initial_cash: float = 10000.0) -> PaperAccount:
    """读取本地模拟账户；如果文件不存在，就创建一个空账户对象。"""
    ledger_path = resolve_ledger_path(path)
    if not ledger_path.exists():
        return PaperAccount(
            cash=float(initial_cash),
            initial_cash=float(initial_cash),
            updated_at=current_timestamp(),
        )

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"模拟盘账户文件格式不正确：{ledger_path}")
    return PaperAccount.from_dict(payload, initial_cash=float(initial_cash))


def save_paper_account(account: PaperAccount, path: str | Path) -> Path:
    """把模拟账户保存到本地 JSON 文件。"""
    ledger_path = resolve_ledger_path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    account.updated_at = current_timestamp()
    payload = {
        "cash": round(account.cash, 4),
        "initial_cash": round(account.initial_cash, 4),
        "updated_at": account.updated_at,
        "positions": {
            symbol: asdict(position)
            for symbol, position in sorted(account.positions.items())
            if position.shares > 0
        },
        "trades": [asdict(trade) for trade in account.trades],
    }
    ledger_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ledger_path


def run_paper_trading_from_result(
    result: Any,
    config: PaperTradingConfig,
    llm_client: Any | None = None,
    execution_price: float | None = None,
) -> dict[str, Any]:
    """根据完整研究结果运行一次模拟盘自动交易。

    执行顺序：

        1. 读取本地模拟账户。
        2. 先尝试复盘过去 pending 的模拟成交。
        3. 根据本次最终信号和风控护栏决定是否模拟下单。
        4. 保存账户。
        5. 返回机器可读摘要，写入 final report 和 full_state.json。
    """
    ledger_path = resolve_ledger_path(config.ledger_path)
    if not config.enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "ledger_path": str(ledger_path),
        }

    account = load_paper_account(ledger_path, initial_cash=config.initial_cash)
    reviewed_count = 0
    if config.review_pending:
        reviewed_count = review_pending_paper_trades(
            account=account,
            holding_days=config.review_holding_days,
            benchmark_symbol=config.benchmark_symbol,
            benchmark_name=config.benchmark_name,
            llm_client=llm_client,
        )

    symbol = normalize_cn_symbol(str(result.final_state.get("symbol", "")))
    trade_date = str(result.final_state.get("trade_date", date.today().strftime("%Y-%m-%d")))

    try:
        price = execution_price or fetch_execution_price(symbol, trade_date)
    except Exception as error:
        save_paper_account(account, ledger_path)
        return {
            "enabled": True,
            "status": "price_unavailable",
            "ledger_path": str(ledger_path),
            "error": str(error),
            "reviewed_pending_count": reviewed_count,
            "account": build_account_summary(account),
        }

    trade = simulate_trade_decision(
        account=account,
        result=result,
        symbol=symbol,
        trade_date=trade_date,
        price=price,
        config=config,
    )
    save_paper_account(account, ledger_path)

    return {
        "enabled": True,
        "status": trade.status,
        "ledger_path": str(ledger_path),
        "execution_price": round(price, 4),
        "reviewed_pending_count": reviewed_count,
        "order": asdict(trade),
        "account": build_account_summary(account),
    }


def simulate_trade_decision(
    account: PaperAccount,
    result: Any,
    symbol: str,
    trade_date: str,
    price: float,
    config: PaperTradingConfig,
) -> PaperTrade:
    """把本次研究结果转换成一笔模拟订单或跳过记录。"""
    action = str(getattr(result.trade_signal, "action", "HOLD") or "HOLD")
    rating = extract_rating(result)
    signal_text = str(getattr(result.trade_signal, "chinese_action", action))

    if action == "BUY":
        trade = execute_buy(
            account=account,
            result=result,
            symbol=symbol,
            trade_date=trade_date,
            price=price,
            config=config,
            rating=rating,
            signal_text=signal_text,
        )
    elif action == "SELL":
        trade = execute_sell(
            account=account,
            result=result,
            symbol=symbol,
            trade_date=trade_date,
            price=price,
            rating=rating,
            signal_text=signal_text,
        )
    else:
        trade = build_skipped_trade(
            account=account,
            symbol=symbol,
            trade_date=trade_date,
            action=action,
            price=price,
            rating=rating,
            signal_text=signal_text,
            reason="最终机器交易信号不是 BUY/SELL，本次模拟盘不下单。",
        )

    # skipped 也写入 trades，是为了以后回看“为什么那天没有动”。
    account.trades.append(trade)
    return trade


def execute_buy(
    account: PaperAccount,
    result: Any,
    symbol: str,
    trade_date: str,
    price: float,
    config: PaperTradingConfig,
    rating: str,
    signal_text: str,
) -> PaperTrade:
    """执行模拟买入。

    买入金额由三层限制共同决定：

        1. 账户现金不能为负；
        2. 单只股票总仓位不能超过全局上限；
        3. 单次加仓不能超过风控护栏 max_single_add_pct。

    A 股买入按 100 股整数倍取整。
    """
    stock_name = resolve_trade_stock_name(result, symbol)
    buy_rule = evaluate_a_share_buy_universe(
        symbol=symbol,
        name=stock_name,
        allowed_universe=config.buy_universe,
    )
    if not buy_rule.allowed_to_buy:
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            f"不属于当前允许买入范围，模拟盘跳过买入。规则：{buy_rule.reason}",
        )

    position = account.positions.get(symbol)
    current_shares = position.shares if position is not None else 0
    current_position_value = current_shares * price

    guardrail = result.risk_guardrail
    if current_shares <= 0 and not bool(getattr(guardrail, "allow_new_position", False)):
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            "风控护栏不允许新开仓，所以模拟盘跳过买入。",
        )

    if current_shares > 0 and not bool(getattr(guardrail, "allow_add_position", False)):
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            "已有持仓，但风控护栏不允许继续加仓，所以模拟盘跳过买入。",
        )

    total_equity = calculate_total_equity(account)
    guardrail_position_pct = max(float(getattr(guardrail, "max_position_pct", 0.0)), 0.0)
    max_position_pct = min(config.max_single_position_pct, guardrail_position_pct)
    max_position_amount = total_equity * max_position_pct
    remaining_position_amount = max_position_amount - current_position_value

    if remaining_position_amount < config.min_trade_amount:
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            "当前持仓已经接近风控上限，或剩余可买金额低于最小成交金额。",
        )

    single_add_pct = max(float(getattr(guardrail, "max_single_add_pct", 0.0)), 0.0)
    single_add_amount = total_equity * single_add_pct
    buy_budget = min(account.cash, remaining_position_amount, single_add_amount)

    if buy_budget < config.min_trade_amount:
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            "可用现金或单次加仓额度低于最小成交金额。",
        )

    shares = round_down_to_lot(buy_budget / price, config.lot_size)
    if shares <= 0:
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "BUY",
            price,
            rating,
            signal_text,
            "按 A 股 100 股整数倍取整后，买入股数为 0。",
        )

    amount = shares * price
    old_cost_amount = (position.avg_cost * position.shares) if position is not None else 0.0
    new_shares = current_shares + shares
    new_avg_cost = (old_cost_amount + amount) / new_shares
    account.cash -= amount
    account.positions[symbol] = build_position(
        symbol=symbol,
        shares=new_shares,
        avg_cost=new_avg_cost,
        price=price,
    )

    return PaperTrade(
        trade_id=build_trade_id(symbol, trade_date, "BUY", len(account.trades) + 1),
        symbol=symbol,
        trade_date=trade_date,
        action="BUY",
        shares=shares,
        price=round(price, 4),
        amount=round(amount, 4),
        cash_after=round(account.cash, 4),
        position_after=new_shares,
        reason=build_order_reason(result, "按最终 BUY 信号和风控仓位上限模拟买入。"),
        status="filled",
        source_rating=rating,
        source_signal=signal_text,
        created_at=current_timestamp(),
        review_status="pending",
    )


def execute_sell(
    account: PaperAccount,
    result: Any,
    symbol: str,
    trade_date: str,
    price: float,
    rating: str,
    signal_text: str,
) -> PaperTrade:
    """执行模拟卖出。

    当前版本不做融券卖空，只处理已有持仓的卖出。
    如果没有持仓，就记录 skipped。
    """
    position = account.positions.get(symbol)
    if position is None or position.shares <= 0:
        return build_skipped_trade(
            account,
            symbol,
            trade_date,
            "SELL",
            price,
            rating,
            signal_text,
            "最终信号偏卖出，但模拟账户没有该股票持仓，所以无需卖出。",
        )

    shares = position.shares
    amount = shares * price
    account.cash += amount
    account.positions.pop(symbol, None)

    return PaperTrade(
        trade_id=build_trade_id(symbol, trade_date, "SELL", len(account.trades) + 1),
        symbol=symbol,
        trade_date=trade_date,
        action="SELL",
        shares=shares,
        price=round(price, 4),
        amount=round(amount, 4),
        cash_after=round(account.cash, 4),
        position_after=0,
        reason=build_order_reason(result, "按最终 SELL 信号模拟清仓卖出。"),
        status="filled",
        source_rating=rating,
        source_signal=signal_text,
        created_at=current_timestamp(),
        review_status="pending",
    )


def build_skipped_trade(
    account: PaperAccount,
    symbol: str,
    trade_date: str,
    action: str,
    price: float,
    rating: str,
    signal_text: str,
    reason: str,
) -> PaperTrade:
    """生成一条未成交记录。"""
    position = account.positions.get(symbol)
    position_after = position.shares if position is not None else 0
    return PaperTrade(
        trade_id=build_trade_id(symbol, trade_date, action, len(account.trades) + 1),
        symbol=symbol,
        trade_date=trade_date,
        action=action,
        shares=0,
        price=round(price, 4),
        amount=0.0,
        cash_after=round(account.cash, 4),
        position_after=position_after,
        reason=reason,
        status="skipped",
        source_rating=rating,
        source_signal=signal_text,
        created_at=current_timestamp(),
        review_status="not_required",
    )


def fetch_execution_price(symbol: str, trade_date: str) -> float:
    """获取模拟成交价格。

    当分析日期是今天时，优先尝试公开实时/近实时行情；
    如果实时行情不可用，再退回最近一个交易日收盘价。

    这不是券商真实成交价，只是模拟盘的可解释近似价格。
    """
    normalized_symbol = normalize_cn_symbol(symbol)
    if trade_date == date.today().strftime("%Y-%m-%d"):
        try:
            quote = route_realtime_quote(normalized_symbol, vendor="auto")
            if quote.latest_price is not None and quote.latest_price > 0:
                return float(quote.latest_price)
        except Exception:
            pass

    end = datetime.strptime(trade_date, "%Y-%m-%d").date()
    start = end - timedelta(days=14)
    history = route_daily_history(
        normalized_symbol,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        vendor="auto",
    )
    if history is None or history.empty:
        raise ValueError(f"{normalized_symbol} 没有可用于模拟成交的历史行情。")

    frame = history.copy()
    frame = frame.dropna(subset=["Close"]).sort_values("Date")
    if frame.empty:
        raise ValueError(f"{normalized_symbol} 历史行情缺少收盘价。")

    price = float(frame["Close"].iloc[-1])
    if price <= 0:
        raise ValueError(f"{normalized_symbol} 模拟成交价无效：{price}")
    return price


def review_pending_paper_trades(
    account: PaperAccount,
    holding_days: int = 5,
    benchmark_symbol: str = "000300",
    benchmark_name: str = "沪深300",
    llm_client: Any | None = None,
) -> int:
    """复盘账户里还没有完成的模拟成交。

    复盘逻辑复用 memory.outcome：
        买入类评级希望后续相对基准为正；
        卖出/低配类评级希望后续相对基准为负。

    如果分析日期太近，行情数量不够，就保持 pending，留到下次再试。
    """
    updated_count = 0
    for trade in account.trades:
        if trade.status != "filled" or trade.review_status != "pending":
            continue

        outcome = resolve_decision_outcome(
            symbol=trade.symbol,
            trade_date=trade.trade_date,
            rating=trade.source_rating,
            final_decision=trade.reason,
            holding_days=holding_days,
            benchmark_symbol=benchmark_symbol,
            benchmark_name=benchmark_name,
            llm_client=llm_client,
        )
        if outcome is None:
            continue

        trade.review_status = "reviewed"
        trade.raw_return = outcome.raw_return
        trade.alpha_return = outcome.alpha_return
        trade.holding_days = outcome.holding_days
        trade.reflection = outcome.reflection
        updated_count += 1

    return updated_count


def build_position(symbol: str, shares: int, avg_cost: float, price: float) -> PaperPosition:
    """根据当前价格生成持仓对象。"""
    market_value = shares * price
    cost_amount = shares * avg_cost
    unrealized_pnl = market_value - cost_amount
    unrealized_return = 0.0 if cost_amount == 0 else unrealized_pnl / cost_amount
    return PaperPosition(
        symbol=symbol,
        shares=int(shares),
        avg_cost=round(avg_cost, 4),
        last_price=round(price, 4),
        market_value=round(market_value, 4),
        unrealized_pnl=round(unrealized_pnl, 4),
        unrealized_return=round(unrealized_return, 6),
        updated_at=current_timestamp(),
    )


def calculate_total_equity(account: PaperAccount) -> float:
    """计算模拟账户总资产。

    持仓有最近市值时用市值；
    如果没有最近市值，就退回成本估算。
    """
    position_value = 0.0
    for position in account.positions.values():
        if position.market_value > 0:
            position_value += position.market_value
        else:
            position_value += position.avg_cost * position.shares
    return account.cash + position_value


def build_account_summary(account: PaperAccount) -> dict[str, Any]:
    """生成账户摘要，写入报告和 full_state。"""
    total_equity = calculate_total_equity(account)
    total_return = 0.0
    if account.initial_cash:
        total_return = (total_equity - account.initial_cash) / account.initial_cash
    return {
        "cash": round(account.cash, 4),
        "initial_cash": round(account.initial_cash, 4),
        "total_equity": round(total_equity, 4),
        "total_return": round(total_return, 6),
        "position_count": len([p for p in account.positions.values() if p.shares > 0]),
        "trade_count": len(account.trades),
        "positions": {
            symbol: asdict(position)
            for symbol, position in sorted(account.positions.items())
            if position.shares > 0
        },
    }


def build_order_reason(result: Any, prefix: str) -> str:
    """把下单依据压缩成一段可读说明。"""
    guardrail_summary = str(getattr(result.risk_guardrail, "chinese_summary", ""))
    executive_summary = str(getattr(result.portfolio_decision, "executive_summary", ""))
    return "\n".join(
        [
            prefix,
            f"风控护栏：{guardrail_summary}",
            f"组合经理摘要：{executive_summary}",
        ]
    )


def extract_rating(result: Any) -> str:
    """从结果对象里提取 Portfolio Manager 评级。"""
    rating = getattr(getattr(result, "portfolio_decision", None), "rating", "")
    return str(getattr(rating, "value", rating) or "")


def resolve_trade_stock_name(result: Any, symbol: str) -> str:
    """尽量拿到股票简称，用于 ST、退市等买入范围判断。

    常规研究结果会把 symbol、trade_date 放在 final_state 里。
    后续如果路由器或用户问题识别阶段把 stock_name 也放进去，
    这里会优先使用它；否则再按股票代码查询名称表。
    """
    final_state = getattr(result, "final_state", {}) or {}
    if isinstance(final_state, dict):
        value = final_state.get("stock_name") or final_state.get("name")
        if value:
            return str(value)

    return find_stock_name_by_symbol(symbol) or symbol


def round_down_to_lot(shares: float, lot_size: int) -> int:
    """按 A 股买入最小单位向下取整。"""
    if lot_size <= 0:
        return int(math.floor(shares))
    return int(math.floor(shares / lot_size) * lot_size)


def build_trade_id(symbol: str, trade_date: str, action: str, sequence: int) -> str:
    """生成可读的模拟成交编号。"""
    return f"{trade_date}-{symbol}-{action}-{sequence:04d}"


def current_timestamp() -> str:
    """当前时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
