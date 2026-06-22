"""A 股交易决策记忆日志。

这个文件参考原版 TradingAgents 的 TradingMemoryLog，
但先实现最核心的第一阶段：

1. 每次完整分析结束后，把最终交易决策写入本地 Markdown 日志。
2. 下次再分析同一只股票时，从日志里读取过去经验。
3. 把过去经验整理成 past_context，注入 Portfolio Manager 的 Prompt。

注意：
    这里不是“数据缓存”。
    数据缓存存的是公司资料、财务表这类外部原材料。

    这里是“交易记忆”。
    它存的是系统自己过去做过什么判断。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from tradingagents_cn.memory.outcome import resolve_decision_outcome
from tradingagents_cn.memory.reflection import ReflectionLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_LOG_PATH = PROJECT_ROOT / "outputs" / "memory" / "trading_memory.md"


@dataclass
class TradingMemoryEntry:
    """一条交易记忆记录。

    date:
        当时分析日期。

    symbol:
        股票代码。

    rating:
        当时 Portfolio Manager 的最终评级。

    pending:
        True 表示这条记录还没有做事后复盘。

    decision:
        当时完整的最终交易决策文本。

    reflection:
        事后反思文本。
        当前阶段先预留字段，下一步接收益回测和大模型反思时会写入。
    """

    date: str
    symbol: str
    rating: str
    pending: bool
    decision: str
    reflection: str = ""
    raw_return: str | None = None
    alpha_return: str | None = None
    holding_days: str | None = None


class TradingMemoryLog:
    """追加式 Markdown 交易记忆日志。

    为什么用 Markdown？
        1. 人可以直接打开看；
        2. 容易调试；
        3. 不需要数据库；
        4. 和原版 TradingAgents 当前设计保持一致。
    """

    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, log_path: str | Path | None = None, max_entries: int | None = None):
        configured_path = (
            log_path
            or os.getenv("TRADINGAGENTS_CN_MEMORY_LOG_PATH")
            or DEFAULT_MEMORY_LOG_PATH
        )
        self.log_path = Path(configured_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

    def store_decision(
        self,
        symbol: str,
        trade_date: str,
        rating: str,
        final_trade_decision: str,
    ) -> None:
        """把本次最终决策写入日志。

        当前先写成 pending，表示这条记录还没有做事后收益复盘。
        后续 Phase B 会把 pending 更新成：
            原始收益、相对基准收益、持有天数、反思文本。
        """
        if self.has_pending_entry(symbol=symbol, trade_date=trade_date):
            return

        tag = f"[{trade_date} | {symbol} | {rating} | pending]"
        entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(entry)

    def has_pending_entry(self, symbol: str, trade_date: str) -> bool:
        """判断同一股票同一日期是否已经有 pending 记录。"""
        if not self.log_path.exists():
            return False

        prefix = f"[{trade_date} | {symbol} |"
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix) and line.endswith("| pending]"):
                return True
        return False

    def load_entries(self) -> list[TradingMemoryEntry]:
        """读取并解析所有记忆记录。"""
        if not self.log_path.exists():
            return []

        text = self.log_path.read_text(encoding="utf-8")
        raw_entries = [
            block.strip()
            for block in text.split(self._SEPARATOR)
            if block.strip()
        ]

        entries: list[TradingMemoryEntry] = []
        for raw in raw_entries:
            entry = self.parse_entry(raw)
            if entry is not None:
                entries.append(entry)
        return entries

    def get_pending_entries(self, symbol: str | None = None) -> list[TradingMemoryEntry]:
        """读取 pending 记录。

        如果传入 symbol，只返回这只股票的 pending 记录。
        """
        entries = [entry for entry in self.load_entries() if entry.pending]
        if symbol is not None:
            entries = [entry for entry in entries if entry.symbol == symbol]
        return entries

    def resolve_pending_outcomes(
        self,
        symbol: str | None = None,
        holding_days: int = 5,
        benchmark_symbol: str = "000300",
        benchmark_name: str = "沪深300",
        llm_client: ReflectionLLMClient | None = None,
    ) -> int:
        """尝试复盘 pending 记忆。

        返回成功更新的记录数量。

        symbol:
            传入股票代码时，只复盘这只股票。
            传入 None 时，复盘日志里所有 pending 记录。

        如果分析日期太近、行情不足、或者接口失败，
        对应 pending 记录会保持原样，下次运行再尝试。
        """
        updates: list[dict[str, object]] = []
        for entry in self.get_pending_entries(symbol=symbol):
            outcome = resolve_decision_outcome(
                symbol=entry.symbol,
                trade_date=entry.date,
                rating=entry.rating,
                final_decision=entry.decision,
                holding_days=holding_days,
                benchmark_symbol=benchmark_symbol,
                benchmark_name=benchmark_name,
                llm_client=llm_client,
            )
            if outcome is None:
                continue

            updates.append(
                {
                    "symbol": entry.symbol,
                    "trade_date": entry.date,
                    "raw_return": outcome.raw_return,
                    "alpha_return": outcome.alpha_return,
                    "holding_days": outcome.holding_days,
                    "reflection": outcome.reflection,
                }
            )

        if updates:
            self.batch_update_with_outcomes(updates)

        return len(updates)

    def batch_update_with_outcomes(self, updates: list[dict[str, object]]) -> None:
        """把 pending 记录批量更新为已复盘记录。"""
        if not updates or not self.log_path.exists():
            return

        update_map = {
            (str(update["trade_date"]), str(update["symbol"])): update
            for update in updates
        }

        text = self.log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)
        new_blocks: list[str] = []

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue

            lines = stripped.splitlines()
            tag_line = lines[0].strip()
            matched = False

            for (trade_date, symbol), update in list(update_map.items()):
                prefix = f"[{trade_date} | {symbol} |"
                if tag_line.startswith(prefix) and tag_line.endswith("| pending]"):
                    fields = [field.strip() for field in tag_line[1:-1].split("|")]
                    rating = fields[2]
                    raw_pct = format_percent(float(update["raw_return"]))
                    alpha_pct = format_percent(float(update["alpha_return"]))
                    new_tag = (
                        f"[{trade_date} | {symbol} | {rating} | "
                        f"{raw_pct} | {alpha_pct} | {update['holding_days']}d]"
                    )
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{update['reflection']}"
                    )
                    del update_map[(trade_date, symbol)]
                    matched = True
                    break

            if not matched:
                new_blocks.append(block)

        new_text = self._SEPARATOR.join(new_blocks)
        temp_path = self.log_path.with_suffix(".tmp")
        temp_path.write_text(new_text, encoding="utf-8")
        temp_path.replace(self.log_path)

    def get_past_context(
        self,
        symbol: str,
        n_same: int = 5,
        n_cross: int = 3,
        include_pending: bool = True,
    ) -> str:
        """生成给 Portfolio Manager 使用的历史经验文本。

        same:
            同一只股票的历史记录，完整展示 decision 和 reflection。

        cross:
            其他股票的历史记录，只展示更短的经验摘要。

        include_pending:
            当前阶段还没有接收益复盘，所以允许 pending 记录也进入上下文。
            等 Phase B 完成后，可以改成只注入已经有 reflection 的记录。
        """
        entries = self.load_entries()
        if not include_pending:
            entries = [entry for entry in entries if not entry.pending]

        same: list[TradingMemoryEntry] = []
        cross: list[TradingMemoryEntry] = []

        for entry in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if entry.symbol == symbol and len(same) < n_same:
                same.append(entry)
            elif entry.symbol != symbol and len(cross) < n_cross:
                cross.append(entry)

        if not same and not cross:
            return ""

        parts: list[str] = []
        if same:
            parts.append(f"同一股票 {symbol} 的历史分析记录（越靠前越新）：")
            parts.extend(self.format_full_entry(entry) for entry in same)

        if cross:
            parts.append("其他股票的近期经验：")
            parts.extend(self.format_short_entry(entry) for entry in cross)

        return "\n\n".join(parts)

    def parse_entry(self, raw: str) -> TradingMemoryEntry | None:
        """把 Markdown 片段解析成 TradingMemoryEntry。"""
        lines = raw.strip().splitlines()
        if not lines:
            return None

        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None

        fields = [field.strip() for field in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None

        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)

        pending = fields[3] == "pending"
        return TradingMemoryEntry(
            date=fields[0],
            symbol=fields[1],
            rating=fields[2],
            pending=pending,
            raw_return=None if pending else fields[3],
            alpha_return=fields[4] if len(fields) > 4 else None,
            holding_days=fields[5] if len(fields) > 5 else None,
            decision=decision_match.group(1).strip() if decision_match else "",
            reflection=reflection_match.group(1).strip() if reflection_match else "",
        )

    def format_full_entry(self, entry: TradingMemoryEntry) -> str:
        """格式化同一股票的完整历史记录。"""
        tag = format_entry_tag(entry)
        parts = [
            tag,
            f"DECISION:\n{entry.decision}",
        ]
        if entry.reflection:
            parts.append(f"REFLECTION:\n{entry.reflection}")
        return "\n\n".join(parts)

    def format_short_entry(self, entry: TradingMemoryEntry) -> str:
        """格式化其他股票的短经验记录。"""
        tag = format_entry_tag(entry)
        if entry.reflection:
            return f"{tag}\n{entry.reflection}"

        summary = entry.decision[:300]
        if len(entry.decision) > 300:
            summary += "..."
        return f"{tag}\n{summary}"


def format_entry_tag(entry: TradingMemoryEntry) -> str:
    """把记忆记录格式化成人可读标签。"""
    if entry.pending:
        return f"[{entry.date} | {entry.symbol} | {entry.rating} | pending]"

    raw = entry.raw_return or "n/a"
    alpha = entry.alpha_return or "n/a"
    holding = entry.holding_days or "n/a"
    return f"[{entry.date} | {entry.symbol} | {entry.rating} | {raw} | {alpha} | {holding}]"


def format_percent(value: float) -> str:
    """把小数收益格式化成百分比。"""
    return f"{value:+.1%}"
