"""完整研究图的 GraphSetup。

原版 TradingAgents 有独立 GraphSetup：
    它不负责跑模型，也不负责取数据，
    只负责把节点和边装配成 LangGraph。

A 股版之前把装图逻辑写在 research_report_state_graph.py 里。
这个文件把装图逻辑抽成类，主图会更像原项目，也更方便以后替换节点。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph


@dataclass(frozen=True)
class AnalystNodeSpec:
    """一个 Analyst 在图里的节点定义。"""

    key: str
    node_name: str
    node_func: Callable[[dict[str, Any]], dict[str, Any]]


class GraphSetup:
    """负责装配 A 股 TradingAgents StateGraph。"""

    def __init__(
        self,
        state_schema: type,
        prepare_node: Callable[[dict[str, Any]], dict[str, Any]],
        analyst_specs: Mapping[str, AnalystNodeSpec],
        summary_node: Callable[[dict[str, Any]], dict[str, Any]],
        bull_node: Callable[[dict[str, Any]], dict[str, Any]],
        bear_node: Callable[[dict[str, Any]], dict[str, Any]],
        research_manager_node: Callable[[dict[str, Any]], dict[str, Any]],
        trader_node: Callable[[dict[str, Any]], dict[str, Any]],
        aggressive_risk_node: Callable[[dict[str, Any]], dict[str, Any]],
        conservative_risk_node: Callable[[dict[str, Any]], dict[str, Any]],
        neutral_risk_node: Callable[[dict[str, Any]], dict[str, Any]],
        portfolio_manager_node: Callable[[dict[str, Any]], dict[str, Any]],
        should_continue_debate: Callable[[dict[str, Any]], str],
        should_continue_risk_analysis: Callable[[dict[str, Any]], str],
    ) -> None:
        self.state_schema = state_schema
        self.prepare_node = prepare_node
        self.analyst_specs = dict(analyst_specs)
        self.summary_node = summary_node
        self.bull_node = bull_node
        self.bear_node = bear_node
        self.research_manager_node = research_manager_node
        self.trader_node = trader_node
        self.aggressive_risk_node = aggressive_risk_node
        self.conservative_risk_node = conservative_risk_node
        self.neutral_risk_node = neutral_risk_node
        self.portfolio_manager_node = portfolio_manager_node
        self.should_continue_debate = should_continue_debate
        self.should_continue_risk_analysis = should_continue_risk_analysis

    def setup_graph(
        self,
        selected_analysts: tuple[str, ...],
        checkpointer: Any | None = None,
    ):
        """装配并编译完整 StateGraph。"""
        workflow = StateGraph(self.state_schema)

        workflow.add_node("prepare_prompts", self.prepare_node)
        for analyst in selected_analysts:
            spec = self.analyst_specs[analyst]
            workflow.add_node(spec.node_name, spec.node_func)

        workflow.add_node("summary_agent", self.summary_node)
        workflow.add_node("bull_researcher", self.bull_node)
        workflow.add_node("bear_researcher", self.bear_node)
        workflow.add_node("research_manager", self.research_manager_node)
        workflow.add_node("trader", self.trader_node)
        workflow.add_node("aggressive_risk_analyst", self.aggressive_risk_node)
        workflow.add_node("conservative_risk_analyst", self.conservative_risk_node)
        workflow.add_node("neutral_risk_analyst", self.neutral_risk_node)
        workflow.add_node("portfolio_manager", self.portfolio_manager_node)

        workflow.set_entry_point("prepare_prompts")
        self._wire_analysts(workflow, selected_analysts)
        self._wire_research_and_risk(workflow)

        return workflow.compile(checkpointer=checkpointer)

    def _wire_analysts(self, workflow: StateGraph, selected_analysts: tuple[str, ...]) -> None:
        """连接前半段 Analyst 节点。"""
        previous_node = "prepare_prompts"
        for analyst in selected_analysts:
            node_name = self.analyst_specs[analyst].node_name
            workflow.add_edge(previous_node, node_name)
            previous_node = node_name
        workflow.add_edge(previous_node, "summary_agent")

    def _wire_research_and_risk(self, workflow: StateGraph) -> None:
        """连接 Summary 之后的辩论、交易和风控节点。"""
        workflow.add_edge("summary_agent", "bull_researcher")
        workflow.add_conditional_edges(
            "bull_researcher",
            self.should_continue_debate,
            {
                "bear_researcher": "bear_researcher",
                "research_manager": "research_manager",
            },
        )
        workflow.add_conditional_edges(
            "bear_researcher",
            self.should_continue_debate,
            {
                "bull_researcher": "bull_researcher",
                "research_manager": "research_manager",
            },
        )

        workflow.add_edge("research_manager", "trader")
        workflow.add_edge("trader", "aggressive_risk_analyst")
        workflow.add_conditional_edges(
            "aggressive_risk_analyst",
            self.should_continue_risk_analysis,
            {
                "conservative_risk_analyst": "conservative_risk_analyst",
                "portfolio_manager": "portfolio_manager",
            },
        )
        workflow.add_conditional_edges(
            "conservative_risk_analyst",
            self.should_continue_risk_analysis,
            {
                "neutral_risk_analyst": "neutral_risk_analyst",
                "portfolio_manager": "portfolio_manager",
            },
        )
        workflow.add_conditional_edges(
            "neutral_risk_analyst",
            self.should_continue_risk_analysis,
            {
                "aggressive_risk_analyst": "aggressive_risk_analyst",
                "portfolio_manager": "portfolio_manager",
            },
        )
        workflow.add_edge("portfolio_manager", END)
