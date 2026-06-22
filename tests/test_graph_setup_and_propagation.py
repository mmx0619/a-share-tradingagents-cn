import unittest

from tradingagents_cn.graph import (
    GraphSetup,
    ResearchGraphPropagator,
    ResearchInputConfig,
)
from tradingagents_cn.graph.research_report_state_graph import (
    ANALYST_NODE_SPECS,
    ResearchReportGraphState,
    aggressive_risk_analyst_node,
    bear_researcher_node,
    bull_researcher_node,
    conservative_risk_analyst_node,
    neutral_risk_analyst_node,
    portfolio_manager_node,
    prepare_prompts_node,
    research_manager_node,
    should_continue_debate,
    should_continue_risk_analysis,
    summary_agent_node,
    trader_node,
)


class GraphSetupAndPropagationTest(unittest.TestCase):
    def test_propagator_should_create_initial_state_and_config(self):
        """Propagator 应负责创建初始 state 和 LangGraph config。"""
        config = ResearchInputConfig(save_full_state=False)
        propagator = ResearchGraphPropagator(max_recur_limit=77)

        state = propagator.create_initial_state(
            symbol="000725",
            trade_date="2026-06-18",
            config=config,
            llm_client=None,
            temperature=0.1,
            max_debate_rounds=2,
            max_risk_discuss_rounds=3,
            enable_memory=True,
            memory_log_path=None,
            selected_analysts=("market",),
        )
        invoke_config = propagator.build_invoke_config("thread-test")

        self.assertEqual("000725", state["symbol"])
        self.assertEqual(("market",), state["selected_analysts"])
        self.assertEqual(77, invoke_config["recursion_limit"])
        self.assertEqual("thread-test", invoke_config["configurable"]["thread_id"])

    def test_graph_setup_should_compile_with_selected_analysts(self):
        """GraphSetup 应能按 selected_analysts 编译主图。"""
        setup = GraphSetup(
            state_schema=ResearchReportGraphState,
            prepare_node=prepare_prompts_node,
            analyst_specs=ANALYST_NODE_SPECS,
            summary_node=summary_agent_node,
            bull_node=bull_researcher_node,
            bear_node=bear_researcher_node,
            research_manager_node=research_manager_node,
            trader_node=trader_node,
            aggressive_risk_node=aggressive_risk_analyst_node,
            conservative_risk_node=conservative_risk_analyst_node,
            neutral_risk_node=neutral_risk_analyst_node,
            portfolio_manager_node=portfolio_manager_node,
            should_continue_debate=should_continue_debate,
            should_continue_risk_analysis=should_continue_risk_analysis,
        )

        app = setup.setup_graph(selected_analysts=("market",), checkpointer=None)

        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
