"""第 26 步：把路由器接到真实节点函数上。

第 25 步已经讲清楚了：

路由器返回节点名
  ↓
执行器根据节点名找到函数
  ↓
函数执行并写入 state
  ↓
再次问路由器下一步去哪

但是第 25 步用的是 mock 节点。
也就是说，它只是演示流程，没有真的获取行情、新闻，也没有真的组装 Agent。

当前文件做第 26 件事：

把第 25 步里的 mock 节点，逐步换成前面已经写好的真实节点。

当前真实流程：

realtime_quote_node
  ↓
market_node
  ↓
news_node
  ↓
summary_node
  ↓
risk_node
  ↓
trader_node
  ↓
done

注意：
这里说的“真实节点”，指的是节点函数会调用前面已经完成的模块。
例如：

- 实时行情节点会调用 step20_realtime_stateful_workflow.py
- 市场分析节点会调用 step18_stateful_workflow.py
- 新闻节点会调用 step18_stateful_workflow.py
- 实时综合节点会调用 step21_realtime_summary_agent.py
- 实时风控节点会调用 step22_realtime_risk_control_agent.py
- 实时交易员节点会调用 step23_realtime_trader_agent.py

为了测试时不浪费 API，
demo 默认使用 provider="mock"。

如果你想真实调用 DeepSeek，
可以在运行前设置环境变量：

LLM_PROVIDER=deepseek
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import step17_agent_state as agent_state
import step18_stateful_workflow as stateful_workflow
import step20_realtime_stateful_workflow as realtime_workflow
import step21_realtime_summary_agent as realtime_summary
import step22_realtime_risk_control_agent as realtime_risk
import step23_realtime_trader_agent as realtime_trader
import step24_agent_router as router


# 节点函数的统一类型。
#
# 每个节点函数都接收同一个 TradingAgentState，
# 然后把自己的执行结果写回这个 state。
#
# 这里和第 25 步保持一致：
#
#   输入：state
#   输出：None
#
# 因为工作流里的状态是“共享状态”，
# 不需要每个节点都返回一个新对象。
NodeHandler = Callable[[agent_state.TradingAgentState], None]


@dataclass
class RealWorkflowConfig:
    """真实路由工作流的配置。

    这个配置对象主要解决一个问题：

    节点函数只接收 state，
    但是真实节点运行时还需要 provider、model、news_max_items 等参数。

    所以我们把这些参数放进 config，
    再用 build_real_node_handlers(config) 生成节点函数字典。

    字段说明：
    - provider：使用哪个大模型平台，例如 mock、deepseek、openai。
    - model：具体模型名称；为 None 时使用该平台默认模型。
    - news_max_items：最多读取多少条新闻。
    """

    provider: str = "mock"
    model: str | None = None
    news_max_items: int = 5


def run_real_realtime_quote_node(state: agent_state.TradingAgentState) -> None:
    """真实实时行情节点。

    读取：
    - state.symbol

    写入：
    - state.realtime_quote_text

    这里直接复用第 20 步里的 run_realtime_quote_node()。
    那个函数内部会继续调用第 19 步的实时行情采集函数。

    这个节点不调用大模型。
    它只负责把实时/近实时行情快照写入 state。
    """
    realtime_workflow.run_realtime_quote_node(state)


def run_real_market_node(
    state: agent_state.TradingAgentState,
    config: RealWorkflowConfig,
) -> None:
    """真实市场分析节点。

    读取：
    - state.symbol
    - state.start_date
    - state.end_date

    写入：
    - state.market_snapshot_text
    - state.market_report

    内部复用第 18 步的 run_market_node()。
    它会继续调用：

    - AKShare 行情获取
    - 技术指标计算
    - 市场快照生成
    - 市场分析师 Agent

    如果 config.provider 是 mock，
    那么大模型部分会用 mock 模型，方便测试。
    """
    stateful_workflow.run_market_node(
        state=state,
        provider=config.provider,
        model=config.model,
    )


def run_real_news_node(
    state: agent_state.TradingAgentState,
    config: RealWorkflowConfig,
) -> None:
    """真实新闻分析节点。

    读取：
    - state.symbol

    写入：
    - state.news_events_text
    - state.news_report

    内部复用第 18 步的 run_news_node()。
    它会继续调用：

    - 个股新闻获取
    - 新闻事件规则提取
    - 新闻 Agent

    news_max_items 用来控制最多读取多少条新闻。
    数字越大，输入给模型的内容越多，成本也可能越高。
    """
    stateful_workflow.run_news_node(
        state=state,
        provider=config.provider,
        model=config.model,
        news_max_items=config.news_max_items,
    )


def run_real_summary_node(
    state: agent_state.TradingAgentState,
    config: RealWorkflowConfig,
) -> None:
    """真实实时综合节点。

    读取：
    - state.realtime_quote_text
    - state.market_report
    - state.news_report

    写入：
    - state.summary_report

    和第 18 步的普通 summary_node 不同，
    这里使用第 21 步的实时行情感知综合 Agent。

    也就是说：
    综合 Agent 不只看市场报告和新闻报告，
    还会一起看实时行情快照。
    """
    result = realtime_summary.run_realtime_summary_from_state(
        state=state,
        provider=config.provider,
        model=config.model,
    )
    state.set_summary_report(
        report=result.summary_text,
        provider=result.provider,
        model=result.model,
    )


def run_real_risk_node(
    state: agent_state.TradingAgentState,
    config: RealWorkflowConfig,
) -> None:
    """真实实时风控节点。

    读取：
    - state.summary_report

    写入：
    - state.risk_report

    这里直接调用第 22 步的实时风控 Agent。

    注意：
    第 22 步还有一个 run_realtime_risk_from_state()，
    它会先重新运行一次实时综合 Agent。

    但在当前路由流程里，
    summary_node 已经把实时综合报告写入 state.summary_report 了。

    所以这里不需要再跑一次 summary，
    直接把 state.summary_report 交给实时风控 Agent 即可。
    """
    if not state.summary_report:
        raise ValueError("缺少 summary_report，不能运行实时风控节点。")

    result = realtime_risk.run_realtime_risk_control_agent(
        symbol=state.symbol,
        realtime_summary_text=state.summary_report,
        provider=config.provider,
        model=config.model,
    )
    state.set_risk_report(
        report=result.risk_report,
        provider=result.provider,
        model=result.model,
    )


def run_real_trader_node(
    state: agent_state.TradingAgentState,
    config: RealWorkflowConfig,
) -> None:
    """真实实时交易员节点。

    读取：
    - state.risk_report

    写入：
    - state.trader_plan

    这里调用第 23 步的实时交易员 Agent。

    交易员 Agent 必须服从风控 Agent 的边界。
    如果风控报告禁止交易，
    交易员不能自己突破限制去给买入建议。
    """
    if not state.risk_report:
        raise ValueError("缺少 risk_report，不能运行实时交易员节点。")

    result = realtime_trader.run_realtime_trader_agent(
        symbol=state.symbol,
        realtime_risk_report=state.risk_report,
        provider=config.provider,
        model=config.model,
    )
    state.set_trader_plan(
        plan=result.trader_plan,
        provider=result.provider,
        model=result.model,
    )


def build_real_node_handlers(config: RealWorkflowConfig) -> dict[str, NodeHandler]:
    """生成真实节点函数字典。

    第 24 步的路由器只返回节点名，例如：

    - realtime_quote_node
    - market_node
    - news_node
    - summary_node
    - risk_node
    - trader_node

    但是 Python 真正执行时，需要的是函数。

    所以这里返回一个字典：

    节点名 -> 节点函数

    例如：

    "market_node" -> market_node_handler

    为什么这里要在函数里面再定义 market_node_handler？

    因为 market_node 运行时需要 config。
    但路由执行器只会调用 handler(state)，不会额外传 config。

    所以我们在这里用内部函数“记住”config。
    这叫闭包，可以简单理解为：

    内部函数可以使用外层函数里的变量。
    """

    def realtime_quote_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_realtime_quote_node(state)

    def market_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_market_node(state, config=config)

    def news_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_news_node(state, config=config)

    def summary_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_summary_node(state, config=config)

    def risk_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_risk_node(state, config=config)

    def trader_node_handler(state: agent_state.TradingAgentState) -> None:
        run_real_trader_node(state, config=config)

    return {
        "realtime_quote_node": realtime_quote_node_handler,
        "market_node": market_node_handler,
        "news_node": news_node_handler,
        "summary_node": summary_node_handler,
        "risk_node": risk_node_handler,
        "trader_node": trader_node_handler,
    }


def run_router_real_workflow(
    symbol: str,
    start_date: str,
    end_date: str,
    config: RealWorkflowConfig,
    max_steps: int = 20,
) -> agent_state.TradingAgentState:
    """运行路由驱动的真实工作流。

    这就是第 25 步的真实节点版本。

    它做的事情是：

    1. 创建空的 TradingAgentState。
    2. 创建真实节点函数字典。
    3. 不断调用第 24 步的 decide_next_node(state)。
    4. 根据路由结果执行真实节点函数。
    5. 直到路由器返回 done。
    """
    state = agent_state.TradingAgentState(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=config.provider,
        model=config.model,
    )
    node_handlers = build_real_node_handlers(config)

    for step_index in range(1, max_steps + 1):
        decision = router.decide_next_node(state)
        print(f"第 {step_index} 步路由结果：{decision.next_node}")
        print(f"原因：{decision.reason}")

        if decision.next_node == "done":
            print("流程结束。")
            return state

        handler = node_handlers.get(decision.next_node)
        if handler is None:
            raise ValueError(f"没有找到节点处理函数：{decision.next_node}")

        handler(state)
        print(f"已执行节点：{decision.next_node}")
        print()

    raise RuntimeError(f"超过最大执行步数 {max_steps}，可能发生了路由死循环。")


def render_router_real_workflow_report(
    state: agent_state.TradingAgentState,
) -> str:
    """把路由真实工作流的最终 state 渲染成报告。"""
    return f"""{agent_state.render_state_summary(state)}

======== 实时行情快照 ========
{state.realtime_quote_text}

======== 日线市场快照 ========
{state.market_snapshot_text}

======== 新闻事件信号 ========
{state.news_events_text}

======== 市场分析师报告 ========
{state.market_report}

======== 新闻 Agent 报告 ========
{state.news_report}

======== 实时综合报告 ========
{state.summary_report}

======== 实时风控报告 ========
{state.risk_report}

======== 实时交易员预案 ========
{state.trader_plan}
"""


def demo_router_real_workflow() -> None:
    """演示路由器如何驱动真实节点。

    默认参数说明：

    - 股票：002361
    - 日期：2026-01-01 到 2026-06-15
    - 模型：mock

    为什么默认用 mock？

    因为这一步的重点是验证“路由 + 真实节点映射”。
    如果默认直接调用 DeepSeek，
    每次测试都会消耗真实 API。

    你后面要切换成 DeepSeek 时，
    设置环境变量 LLM_PROVIDER=deepseek 即可。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    provider = os.environ.get("LLM_PROVIDER", "mock")
    model = os.environ.get("LLM_MODEL") or None
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "3"))

    config = RealWorkflowConfig(
        provider=provider,
        model=model,
        news_max_items=news_max_items,
    )
    state = run_router_real_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    print(render_router_real_workflow_report(state))


if __name__ == "__main__":
    demo_router_real_workflow()
