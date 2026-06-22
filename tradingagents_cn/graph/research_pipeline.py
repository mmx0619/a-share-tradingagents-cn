"""A 股研究流程正式入口。

这个文件把前面已经完成的两层真正接起来：

    1. research_inputs.py
       负责从 AKShare 等数据源获取真实 A 股数据，
       并把数据整理成 LangGraph 可以接收的初始 state。

    2. research_workflow.py
       负责定义 LangGraph 节点顺序：
           market -> news -> fundamentals

也就是说，这个文件做的是：

    股票代码
      -> 准备真实数据
      -> 执行 LangGraph 研究工作流
      -> 返回包含三个 Agent Prompt 的最终 state

注意：
    当前这一步仍然不调用大模型。
    它只是完成“真实数据进入多 Agent 图”的正式闭环。

为什么先不调用大模型？
    因为 TradingAgents 这类项目要分层调试：
        数据是否正确；
        Prompt 是否正确；
        LangGraph 节点是否正确流转；
        大模型是否正确输出。

    如果这些步骤混在一起，出错时很难判断到底是：
        数据接口失败；
        Prompt 拼错；
        LangGraph 状态没传下去；
        还是大模型返回格式不稳定。
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents_cn.graph.research_inputs import (
    ResearchInputConfig,
    build_research_initial_state,
)
from tradingagents_cn.graph.research_workflow import (
    ResearchWorkflowState,
    build_research_workflow,
)


@dataclass
class ResearchPromptPipelineResult:
    """研究 Prompt 流程的返回结果。

    这个对象是为了让调用方更容易理解最终结果。

    final_state:
        LangGraph 执行完成后的完整状态。
        里面会包含：
            market_prompt
            news_prompt
            fundamentals_prompt
            market_snapshot
            news_materials
            fundamentals_materials
            data_errors

    使用方式示例：

        result = run_research_prompt_pipeline("600519")
        print(result.market_prompt)
        print(result.news_prompt)
        print(result.fundamentals_prompt)

    注意：
        这里的 market_prompt/news_prompt/fundamentals_prompt
        还只是“准备给大模型看的提示词”，不是大模型生成的最终分析报告。
    """

    final_state: ResearchWorkflowState

    @property
    def market_prompt(self) -> str:
        """返回 Market Agent 的完整提示词。"""
        return self.final_state.get("market_prompt", "")

    @property
    def news_prompt(self) -> str:
        """返回 News Agent 的完整提示词。"""
        return self.final_state.get("news_prompt", "")

    @property
    def fundamentals_prompt(self) -> str:
        """返回 Fundamentals Agent 的完整提示词。"""
        return self.final_state.get("fundamentals_prompt", "")

    @property
    def data_errors(self) -> list[str]:
        """返回数据准备阶段的非致命错误。

        例如：
            实时行情接口临时失败；
            新闻接口没有返回数据；
            某张财务表接口超时。

        这些错误不会阻止 LangGraph 继续运行，
        但后续生成报告时应该把它们展示出来，
        避免用户误以为所有数据都完整。
        """
        return self.final_state.get("data_errors", [])


def run_research_prompt_pipeline(
    symbol: str,
    trade_date: str | None = None,
    config: ResearchInputConfig | None = None,
) -> ResearchPromptPipelineResult:
    """运行正式的 A 股研究 Prompt 流程。

    参数：
        symbol:
            A 股股票代码，例如：
                600519
                002361

        trade_date:
            分析日期，格式是 YYYY-MM-DD。
            如果不传，就使用电脑当前日期。

        config:
            数据准备配置。
            可以控制是否拉实时行情、新闻、基本面等。

    执行流程：
        1. 调用 build_research_initial_state()
           获取真实 A 股数据，并组装成初始 state。

        2. 调用 build_research_workflow()
           构建 LangGraph 工作流。

        3. 调用 app.invoke(initial_state)
           让 state 依次经过：
               market 节点
               news 节点
               fundamentals 节点

        4. 返回最终 state。

    返回：
        ResearchPromptPipelineResult。

    重要理解：
        app.invoke(...) 才是真正让 LangGraph 开始运行的语句。
        build_research_workflow() 只是“搭好图”，还没有执行。
    """
    initial_state = build_research_initial_state(
        symbol=symbol,
        trade_date=trade_date,
        config=config,
    )

    app = build_research_workflow()
    final_state = app.invoke(initial_state)

    return ResearchPromptPipelineResult(final_state=final_state)
