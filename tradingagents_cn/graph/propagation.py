"""StateGraph 状态传播工具。

原版 TradingAgents 把“初始 state 怎么建、图怎么 invoke”放在 Propagator。
A 股版之前这些逻辑散在 run_research_report_state_graph(...) 里。

这个文件把它们抽出来：
    1. create_initial_state：创建图的初始状态；
    2. build_invoke_config：创建 LangGraph 调用配置；
    3. select_input_state：根据 resume 决定传初始 state 还是 None。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents_cn.graph.checkpointing import (
    build_thread_config,
    has_checkpoint_for_thread,
)
from tradingagents_cn.graph.research_inputs import ResearchInputConfig


@dataclass
class ResearchGraphPropagator:
    """负责让研究图从初始 state 向最终节点传播。"""

    max_recur_limit: int = 100

    def create_initial_state(
        self,
        symbol: str,
        trade_date: str | None,
        config: ResearchInputConfig,
        llm_client: Any | None,
        temperature: float,
        max_debate_rounds: int,
        max_risk_discuss_rounds: int,
        enable_memory: bool,
        memory_log_path: str | None,
        selected_analysts: tuple[str, ...],
    ) -> dict[str, Any]:
        """创建 StateGraph 的初始状态。

        注意：
            这里不提前取行情、新闻、财报。
            正式 Tool Calling 图会让各 Analyst 自己调用工具。
        """
        return {
            "symbol": symbol,
            "trade_date": trade_date,
            "config": config,
            "llm_client": llm_client,
            "temperature": temperature,
            "max_debate_rounds": max_debate_rounds,
            "max_risk_discuss_rounds": max_risk_discuss_rounds,
            "enable_memory": enable_memory,
            "memory_log_path": memory_log_path,
            "selected_analysts": selected_analysts,
        }

    def build_invoke_config(self, thread_id: str) -> dict[str, Any]:
        """创建 LangGraph invoke 使用的 config。"""
        return {
            **build_thread_config(thread_id),
            "recursion_limit": self.max_recur_limit,
        }

    def select_input_state(
        self,
        initial_state: dict[str, Any],
        thread_id: str,
        resume: bool,
        checkpoint_db_path: str | None = None,
    ) -> dict[str, Any] | None:
        """根据 resume 和 checkpoint 是否存在决定传入什么 state。

        LangGraph 续跑时，传 None 表示从已有 checkpoint 恢复。
        如果没有 checkpoint，就必须传初始 state。
        """
        if resume and has_checkpoint_for_thread(thread_id, checkpoint_db_path):
            return None
        return initial_state
