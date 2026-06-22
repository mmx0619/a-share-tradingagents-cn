"""第 17 步：定义多 Agent 共享状态。

前面你已经理解了：

上一个 Agent 的输出
  ↓
放进下一个 Agent 的 Prompt
  ↓
下一个 Agent 继续输出

这个理解是对的。

但如果项目继续变大，不能一直靠函数参数传一堆字符串。
更工程化的做法是：

所有 Agent 共同读写一个“状态对象”。

比如：

市场分析师 Agent 写入 market_report
新闻 Agent 写入 news_report
综合 Agent 读取 market_report + news_report，写入 summary_report
风控 Agent 读取 summary_report，写入 risk_report
交易员 Agent 读取 risk_report，写入 trader_plan

这个状态对象就是后面接 LangGraph 时最核心的东西。
LangGraph 里的 State，本质上也是类似这种“共享状态”。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    """记录某个 Agent 的一次输出。

    字段说明：
    - agent_name：Agent 名称，比如 market_agent、news_agent。
    - content：Agent 输出的正文。
    - provider：使用的大模型平台，比如 deepseek。
    - model：具体模型名称。

    为什么要记录 provider 和 model：
    后面排查问题时，需要知道这段内容是谁生成的、用的哪个模型。
    """

    agent_name: str
    content: str
    provider: str | None = None
    model: str | None = None


@dataclass
class TradingAgentState:
    """多 Agent 投研流程的共享状态。

    你可以把它理解成一张“流转单”：

    每个 Agent 做完自己的工作，
    就把结果填到这张表里。

    后面的 Agent 不需要重新找前面所有文件，
    只需要从 state 里读取自己需要的字段。
    """

    # 基础输入
    symbol: str
    start_date: str
    end_date: str

    # 原始数据和中间结果
    realtime_quote_text: str | None = None
    market_snapshot_text: str | None = None
    news_events_text: str | None = None

    # 各个 Agent 的报告
    market_report: str | None = None
    news_report: str | None = None
    summary_report: str | None = None
    risk_report: str | None = None
    trader_plan: str | None = None

    # 运行信息
    provider: str | None = None
    model: str | None = None

    # 所有 Agent 的输出记录
    messages: list[AgentMessage] = field(default_factory=list)

    def add_message(
        self,
        agent_name: str,
        content: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """向状态里追加一条 Agent 输出记录。"""
        self.messages.append(
            AgentMessage(
                agent_name=agent_name,
                content=content,
                provider=provider,
                model=model,
            )
        )

    def set_market_report(
        self,
        report: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """保存市场分析师 Agent 的输出。"""
        self.market_report = report
        self.provider = provider or self.provider
        self.model = model or self.model
        self.add_message("market_agent", report, provider, model)

    def set_news_report(
        self,
        report: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """保存新闻 Agent 的输出。"""
        self.news_report = report
        self.provider = provider or self.provider
        self.model = model or self.model
        self.add_message("news_agent", report, provider, model)

    def set_summary_report(
        self,
        report: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """保存综合投研汇总 Agent 的输出。"""
        self.summary_report = report
        self.provider = provider or self.provider
        self.model = model or self.model
        self.add_message("summary_agent", report, provider, model)

    def set_risk_report(
        self,
        report: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """保存风控 Agent 的输出。"""
        self.risk_report = report
        self.provider = provider or self.provider
        self.model = model or self.model
        self.add_message("risk_agent", report, provider, model)

    def set_trader_plan(
        self,
        plan: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """保存交易员 Agent 的输出。"""
        self.trader_plan = plan
        self.provider = provider or self.provider
        self.model = model or self.model
        self.add_message("trader_agent", plan, provider, model)


def render_state_summary(state: TradingAgentState) -> str:
    """把共享状态渲染成方便阅读的摘要。"""
    message_lines = []
    for index, message in enumerate(state.messages, start=1):
        message_lines.append(
            f"{index}. {message.agent_name} "
            f"provider={message.provider} model={message.model}"
        )

    messages_text = "\n".join(message_lines) if message_lines else "暂无 Agent 输出"

    return f"""股票代码：{state.symbol}
行情区间：{state.start_date} 至 {state.end_date}
模型来源：{state.provider}
模型名称：{state.model}

是否已有市场报告：{"是" if state.market_report else "否"}
是否已有新闻报告：{"是" if state.news_report else "否"}
是否已有综合报告：{"是" if state.summary_report else "否"}
是否已有风控报告：{"是" if state.risk_report else "否"}
是否已有交易预案：{"是" if state.trader_plan else "否"}
是否已有实时行情：{"是" if state.realtime_quote_text else "否"}

Agent 输出记录：
{messages_text}
"""


def demo_agent_state() -> None:
    """演示共享状态如何保存多个 Agent 的输出。"""
    state = TradingAgentState(
        symbol="002361",
        start_date="2026-01-01",
        end_date="2026-06-15",
    )

    state.market_snapshot_text = "这里保存市场快照文本。"
    state.realtime_quote_text = "这里保存实时行情快照文本。"
    state.news_events_text = "这里保存新闻事件信号文本。"

    state.set_market_report(
        report="这里保存市场分析师 Agent 的报告。",
        provider="deepseek",
        model="deepseek-v4-pro",
    )
    state.set_news_report(
        report="这里保存新闻 Agent 的报告。",
        provider="deepseek",
        model="deepseek-v4-pro",
    )
    state.set_summary_report(
        report="这里保存综合 Agent 的报告。",
        provider="deepseek",
        model="deepseek-v4-pro",
    )
    state.set_risk_report(
        report="这里保存风控 Agent 的报告。",
        provider="deepseek",
        model="deepseek-v4-pro",
    )
    state.set_trader_plan(
        plan="这里保存交易员 Agent 的条件式预案。",
        provider="deepseek",
        model="deepseek-v4-pro",
    )

    print(render_state_summary(state))


if __name__ == "__main__":
    demo_agent_state()
