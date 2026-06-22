"""完整运行状态日志。

最终 Markdown 报告是给人读的。
full_state.json 是给调试、复盘、前端页面或后续自动化程序读的。

它会保存：

    - 股票、日期、运行时间；
    - selected_analysts；
    - 各 Agent 报告；
    - 多空辩论、风控辩论；
    - Research Manager / Trader / Portfolio Manager 的结构化输出；
    - messages_by_agent 调试消息。
    - paper_trading 模拟盘自动交易结果。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tradingagents_cn.dataflows.stock_directory import find_stock_name_by_symbol


def save_full_state_json(result: Any, output_dir: str | Path = "outputs/run_states") -> Path:
    """保存完整运行状态 JSON，并返回文件路径。"""
    final_state = result.final_state
    symbol = str(final_state.get("symbol", "unknown"))
    trade_date = str(final_state.get("trade_date", "unknown-date"))
    stock_name = find_stock_name_by_symbol(symbol) or symbol
    run_dir_name = f"{sanitize_path_part(stock_name)}_{sanitize_path_part(symbol)}_{sanitize_path_part(trade_date)}"
    path = Path(output_dir) / run_dir_name / "full_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_full_state_payload(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def build_full_state_payload(result: Any) -> dict[str, Any]:
    """构造可 JSON 序列化的完整状态字典。"""
    final_state = result.final_state
    symbol = str(final_state.get("symbol", "unknown"))
    stock_name = find_stock_name_by_symbol(symbol) or symbol

    return {
        "metadata": {
            "symbol": symbol,
            "stock_name": stock_name,
            "trade_date": final_state.get("trade_date"),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selected_analysts": list(getattr(result, "selected_analysts", ()) or ()),
        },
        "final_state": to_jsonable(final_state),
        "reports": {
            "market_report": getattr(result, "market_report", ""),
            "sentiment_report": getattr(result, "sentiment_report", ""),
            "news_report": getattr(result, "news_report", ""),
            "fundamentals_report": getattr(result, "fundamentals_report", ""),
            "summary_report": getattr(result, "summary_report", ""),
        },
        "debate": {
            "max_debate_rounds": getattr(result, "max_debate_rounds", None),
            "bull_argument": getattr(result, "bull_argument", ""),
            "bear_argument": getattr(result, "bear_argument", ""),
            "debate_history": getattr(result, "debate_history", ""),
        },
        "decisions": {
            "research_plan": to_jsonable(getattr(result, "research_plan", None)),
            "investment_plan": getattr(result, "investment_plan", ""),
            "trader_proposal": to_jsonable(getattr(result, "trader_proposal", None)),
            "trader_plan": getattr(result, "trader_plan", ""),
            "portfolio_decision": to_jsonable(getattr(result, "portfolio_decision", None)),
            "final_trade_decision": getattr(result, "final_trade_decision", ""),
            "trade_signal": to_jsonable(getattr(result, "trade_signal", None)),
            "risk_guardrail": to_jsonable(getattr(result, "risk_guardrail", None)),
        },
        "risk_debate": {
            "risk_debate_history": getattr(result, "risk_debate_history", ""),
            "aggressive_risk_argument": getattr(result, "aggressive_risk_argument", ""),
            "conservative_risk_argument": getattr(result, "conservative_risk_argument", ""),
            "neutral_risk_argument": getattr(result, "neutral_risk_argument", ""),
        },
        "messages_by_agent": to_jsonable(getattr(result, "messages_by_agent", {})),
        "tool_call_trace": to_jsonable(getattr(result, "tool_call_trace", [])),
        "tool_call_stats": to_jsonable(getattr(result, "tool_call_stats", {})),
        "reflection_summary": to_jsonable(getattr(result, "reflection_summary", {})),
        "paper_trading": to_jsonable(getattr(result, "paper_trading_result", {})),
        "data_quality_issues": extract_data_quality_issues(
            getattr(result, "messages_by_agent", {})
        ),
        "data_errors": to_jsonable(getattr(result, "data_errors", [])),
    }


def extract_data_quality_issues(messages_by_agent: dict[str, Any]) -> list[dict[str, str]]:
    """从工具返回内容中提取数据质量提示。

    工具结果会被保存在 messages_by_agent 里。
    这里把其中包含“数据质量提示”的片段提取出来，
    方便 full_state.json 的使用者不用翻完整工具文本。
    """
    issues: list[dict[str, str]] = []
    for agent, messages in (messages_by_agent or {}).items():
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            extracted = extract_quality_lines(content)
            if extracted:
                issues.append(
                    {
                        "agent": str(agent),
                        "tool_call_id": str(message.get("tool_call_id") or ""),
                        "issues": "\n".join(extracted),
                    }
                )
    return issues


def extract_quality_lines(content: str) -> list[str]:
    """提取工具文本中的数据质量提示行。"""
    lines = str(content or "").splitlines()
    result: list[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("数据质量提示"):
            collecting = True
            result.append(stripped)
            continue
        if collecting:
            if not stripped:
                break
            if stripped.startswith("- "):
                result.append(stripped)
            else:
                break
    return result


def to_jsonable(value: Any) -> Any:
    """把复杂 Python 对象转换成 JSON 能保存的类型。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump())

    if is_dataclass(value):
        return to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    return str(value)


def sanitize_path_part(text: str) -> str:
    """清理 Windows 路径中不能使用的字符。"""
    cleaned = str(text or "").strip()
    for char in '<>:"/\\|?*':
        cleaned = cleaned.replace(char, "_")
    return cleaned or "unknown"
