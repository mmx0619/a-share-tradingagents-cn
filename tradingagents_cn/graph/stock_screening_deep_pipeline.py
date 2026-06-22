"""候选股深度筛选流程。

第 94-96 步完成的是：

    全市场快照 -> 候选股票池 -> Stock Screening Agent

这一步继续往后接：

    候选股票池 -> 逐个跑单股完整 TradingAgents 分析 -> 按最终信号排序

为什么不直接放进普通“推荐股票”入口？
    因为每只候选股都要跑完整单股链路，会调用多次大模型和多个数据源。
    如果默认对 20 只候选全部深度分析，会很慢也很费额度。

所以这里先做成独立函数，需要时再调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.graph.research_report_pipeline import (
    ChatModelClient,
    ResearchReportPipelineResult,
    run_research_report_pipeline,
)


@dataclass
class DeepScreeningItem:
    """单只候选股的深度分析结果。"""

    symbol: str
    name: str
    screening_reason: str
    action: str
    rating: str
    chinese_action: str
    executive_summary: str


@dataclass
class DeepScreeningResult:
    """深度筛选结果。"""

    items: list[DeepScreeningItem]
    errors: list[str]


ResearchRunner = Callable[..., ResearchReportPipelineResult]


def run_deep_stock_screening(
    candidates: pd.DataFrame,
    trade_date: str | None = None,
    top_n: int = 3,
    config: ResearchInputConfig | None = None,
    llm_client: ChatModelClient | None = None,
    temperature: float = 0.2,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
    runner: ResearchRunner = run_research_report_pipeline,
) -> DeepScreeningResult:
    """对候选股逐个执行完整单股分析，并输出排序后的结果。

    参数：
        candidates:
            stock_screening.py 生成的候选股票池。

        top_n:
            最多深度分析多少只。
            默认 3，避免一次性调用太多大模型。

        runner:
            单股分析函数。
            默认是真实 run_research_report_pipeline。
            测试时可以传入假函数，避免真实联网和真实模型调用。
    """
    actual_trade_date = trade_date or date.today().strftime("%Y-%m-%d")
    selected = select_candidates_for_deep_screening(candidates, top_n=top_n)

    items: list[DeepScreeningItem] = []
    errors: list[str] = []

    for _, row in selected.iterrows():
        symbol = str(row.get("Symbol") or "").strip()
        name = str(row.get("Name") or symbol).strip()
        if not symbol:
            continue

        try:
            result = runner(
                symbol=symbol,
                trade_date=actual_trade_date,
                config=config,
                llm_client=llm_client,
                temperature=temperature,
                max_debate_rounds=max_debate_rounds,
                max_risk_discuss_rounds=max_risk_discuss_rounds,
            )
            items.append(
                DeepScreeningItem(
                    symbol=symbol,
                    name=name,
                    screening_reason=build_screening_reason(row),
                    action=result.trade_signal.action,
                    rating=result.portfolio_decision.rating.value,
                    chinese_action=result.trade_signal.chinese_action,
                    executive_summary=result.portfolio_decision.executive_summary,
                )
            )
        except Exception as error:
            errors.append(f"{name}（{symbol}）深度分析失败：{error}")

    return DeepScreeningResult(
        items=sort_deep_screening_items(items),
        errors=errors,
    )


def select_candidates_for_deep_screening(candidates: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """选择需要进入完整单股分析的前 N 只候选。"""
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    return candidates.head(max(1, int(top_n))).copy()


def build_screening_reason(row: pd.Series) -> str:
    """把候选池原始字段整理成一句入选原因。"""
    parts = []
    for field, label in [
        ("ChangePct", "涨跌幅"),
        ("Amount", "成交额"),
        ("TurnoverRate", "换手率"),
        ("VolumeRatio", "量比"),
        ("Sector", "板块"),
        ("SectorChangePct", "板块涨跌幅"),
        ("DynamicPE", "动态市盈率"),
        ("TotalMarketCap", "总市值"),
    ]:
        value = row.get(field)
        if value is not None and str(value) not in {"", "nan", "None"}:
            parts.append(f"{label}={value}")

    return "；".join(parts) if parts else "来自候选池排序。"


def sort_deep_screening_items(items: list[DeepScreeningItem]) -> list[DeepScreeningItem]:
    """按最终交易信号排序。

    BUY 优先，其次 HOLD，最后 SELL。
    同一档保持原候选顺序。
    """
    action_rank = {
        "BUY": 0,
        "HOLD": 1,
        "SELL": 2,
    }
    return sorted(
        items,
        key=lambda item: action_rank.get(item.action, 99),
    )


def render_deep_screening_result(result: DeepScreeningResult) -> str:
    """把深度筛选结果渲染成人可读 Markdown。"""
    lines = [
        "# A 股候选股深度分析排序",
        "",
        "说明：以下排序来自候选股逐个运行完整单股 TradingAgents 分析后的结果。",
        "",
    ]

    if not result.items:
        lines.append("暂无成功完成深度分析的候选股。")
    else:
        lines.extend(
            [
                "| 排名 | 股票 | 信号 | 评级 | 初筛理由 | 最终摘要 |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for index, item in enumerate(result.items, start=1):
            lines.append(
                f"| {index} | {item.name}（{item.symbol}） | "
                f"{item.action} / {item.chinese_action} | {item.rating} | "
                f"{item.screening_reason} | {item.executive_summary} |"
            )

    if result.errors:
        lines.extend(["", "## 失败记录", ""])
        lines.extend(f"- {error}" for error in result.errors)

    return "\n".join(lines)
