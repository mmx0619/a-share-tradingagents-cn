"""TradingAgentsCNGraph 统一封装。

原项目通常会有一个总控类负责：

    - 创建 LLM；
    - 保存配置；
    - 构建图；
    - 运行单股分析；
    - 从用户自然语言问题触发分析。

这个类就是 A 股版的统一入口。
日常使用时，你不需要直接调用底层一堆函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tradingagents_cn.dataflows.vendor_config import normalize_selected_analysts
from tradingagents_cn.graph.research_inputs import ResearchInputConfig
from tradingagents_cn.graph.research_report_pipeline import (
    ChatModelClient,
    ResearchReportPipelineResult,
)
from tradingagents_cn.graph.research_report_state_graph import run_research_report_state_graph
from tradingagents_cn.graph.user_question_pipeline import (
    UserQuestionPipelineResult,
    run_user_question_pipeline,
)
from tradingagents_cn.llm.factory import create_chat_client


@dataclass
class TradingAgentsCNGraph:
    """A 股 TradingAgents 主图封装类。

    参数：
        selected_analysts:
            本次启用哪些 Analyst。
            默认 market、sentiment、news、fundamentals 全部启用。

        config:
            研究配置。
            如果不传，会创建默认 ResearchInputConfig。

        llm_client:
            已创建好的大模型客户端。
            如果不传，会按 provider/model 创建。

        provider / model:
            模型服务商和模型名。
            支持 deepseek、openai、kimi、gemini。
    """

    selected_analysts: tuple[str, ...] | None = None
    config: ResearchInputConfig | None = None
    llm_client: ChatModelClient | None = None
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    enable_memory: bool = True
    memory_log_path: str | None = None
    checkpoint_db_path: str | None = None
    use_sqlite_checkpoint: bool = True

    def __post_init__(self) -> None:
        """初始化配置和模型客户端。"""
        self.config = self.config or ResearchInputConfig()
        selected = self.selected_analysts or self.config.selected_analysts
        self.selected_analysts = normalize_selected_analysts(selected)
        self.config.selected_analysts = self.selected_analysts
        if self.llm_client is None:
            self.llm_client = create_chat_client(provider=self.provider, model=self.model)

        self.curr_result: ResearchReportPipelineResult | None = None
        self.curr_question_result: UserQuestionPipelineResult | None = None

    def propagate(
        self,
        symbol: str,
        trade_date: str | None = None,
        thread_id: str | None = None,
        resume: bool = False,
        **overrides: Any,
    ) -> ResearchReportPipelineResult:
        """运行单只股票完整研究链路。

        这个名字对齐原项目习惯：
            propagate 表示让状态沿图向前传播，直到最终 Portfolio Manager。
        """
        actual_trade_date = trade_date or date.today().strftime("%Y-%m-%d")
        result = run_research_report_state_graph(
            symbol=symbol,
            trade_date=actual_trade_date,
            config=self.config,
            llm_client=self.llm_client,
            temperature=overrides.get("temperature", self.temperature),
            max_debate_rounds=overrides.get("max_debate_rounds", self.max_debate_rounds),
            max_risk_discuss_rounds=overrides.get(
                "max_risk_discuss_rounds",
                self.max_risk_discuss_rounds,
            ),
            enable_memory=overrides.get("enable_memory", self.enable_memory),
            memory_log_path=overrides.get("memory_log_path", self.memory_log_path),
            thread_id=thread_id,
            checkpoint_db_path=overrides.get("checkpoint_db_path", self.checkpoint_db_path),
            resume=resume,
            use_sqlite_checkpoint=overrides.get(
                "use_sqlite_checkpoint",
                self.use_sqlite_checkpoint,
            ),
            selected_analysts=self.selected_analysts,
        )
        self.curr_result = result
        return result

    def analyze_question(
        self,
        question: str,
        trade_date: str | None = None,
        output_dir: str | Path = "outputs/user_questions",
        thread_id: str | None = None,
        resume: bool = False,
        **overrides: Any,
    ) -> UserQuestionPipelineResult:
        """从用户自然语言问题运行分析链路。"""
        result = run_user_question_pipeline(
            question=question,
            trade_date=trade_date,
            config=self.config,
            llm_client=self.llm_client,
            temperature=overrides.get("temperature", self.temperature),
            max_debate_rounds=overrides.get("max_debate_rounds", self.max_debate_rounds),
            max_risk_discuss_rounds=overrides.get(
                "max_risk_discuss_rounds",
                self.max_risk_discuss_rounds,
            ),
            output_dir=output_dir,
            use_llm_router=overrides.get("use_llm_router", True),
            enable_deep_stock_screening=overrides.get("enable_deep_stock_screening", True),
            deep_stock_screening_top_n=overrides.get("deep_stock_screening_top_n", 3),
            thread_id=thread_id,
            resume=resume,
            checkpoint_db_path=overrides.get("checkpoint_db_path", self.checkpoint_db_path),
        )
        self.curr_question_result = result
        self.curr_result = result.research_result
        return result

    # run 是 propagate 的别名，方便不同使用习惯的人调用。
    run = propagate
