"""第 27 步：使用真正的 LangGraph 运行真实多 Agent 工作流。

第 26 步我们自己写了一个循环：

while 没有结束：
    问路由器下一步去哪
    根据节点名找到函数
    执行函数
    节点写入 state

这一步开始使用真正的 LangGraph。

你可以把 LangGraph 理解成：

它帮我们管理“节点”和“边”。

也就是说：

- 我们告诉 LangGraph 有哪些节点。
- 我们告诉 LangGraph 如何判断下一条边。
- LangGraph 负责按照图结构推动流程运行。

当前文件仍然复用前面已经写好的真实节点：

- realtime_quote_node：实时行情节点
- market_node：市场分析节点
- news_node：新闻分析节点
- summary_node：实时综合节点
- risk_node：实时风控节点
- trader_node：实时交易员节点

这一版和第 26 步的最大区别：

第 26 步：
    我们自己写 for 循环推进流程。

第 27 步：
    LangGraph 负责推进流程。
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import step17_agent_state as agent_state
import step24_agent_router as router
import step26_router_real_workflow as real_workflow


class LangGraphWorkflowState(TypedDict):
    """LangGraph 图里的状态类型。

    注意：
    这里的 state 不是直接等于 TradingAgentState。

    我们在 LangGraph 里包了一层字典：

    {
        "agent_state": TradingAgentState(...),
        "config": RealWorkflowConfig(...),
        "step_log": [...]
    }

    为什么要这样包一层？

    因为真实节点运行时不仅需要 TradingAgentState，
    还需要 provider、model、news_max_items 等配置。

    TradingAgentState 保存业务数据。
    RealWorkflowConfig 保存运行配置。
    step_log 保存每次路由和节点执行记录，方便你观察流程。
    """

    agent_state: agent_state.TradingAgentState
    config: real_workflow.RealWorkflowConfig
    step_log: list[str]


def append_log(
    graph_state: LangGraphWorkflowState,
    message: str,
) -> LangGraphWorkflowState:
    """给 LangGraph 状态追加一条日志。

    LangGraph 节点函数通常返回一个字典，
    表示要更新哪些状态字段。

    这里为了简单清楚，
    每次都返回完整的三个字段：

    - agent_state
    - config
    - step_log
    """
    old_log = graph_state.get("step_log", [])
    new_log = [*old_log, message]
    return {
        "agent_state": graph_state["agent_state"],
        "config": graph_state["config"],
        "step_log": new_log,
    }


def route_next_node(graph_state: LangGraphWorkflowState) -> str:
    """LangGraph 使用的路由函数。

    这个函数是第 24 步路由器和 LangGraph 的连接点。

    LangGraph 会在以下时机调用它：

    - 图刚开始时。
    - 每个节点执行完成后。

    它做的事情是：

    1. 从 graph_state 里取出真正的 TradingAgentState。
    2. 调用第 24 步的 decide_next_node(agent_state)。
    3. 返回下一步节点名。

    返回值必须能匹配 add_conditional_edges() 里的 path_map。

    例如：

    - 返回 "market_node"：LangGraph 就进入 market_node。
    - 返回 "done"：LangGraph 就进入 END，流程结束。
    """
    decision = router.decide_next_node(graph_state["agent_state"])
    print(f"LangGraph 路由结果：{decision.next_node}")
    print(f"原因：{decision.reason}")
    return decision.next_node


def realtime_quote_node(
    graph_state: LangGraphWorkflowState,
) -> LangGraphWorkflowState:
    """LangGraph 实时行情节点。

    LangGraph 会把整个 graph_state 传进来。

    但真正干活的函数只需要 TradingAgentState，
    所以这里取出 graph_state["agent_state"]，
    然后调用第 26 步封装好的真实节点函数。
    """
    state = graph_state["agent_state"]
    real_workflow.run_real_realtime_quote_node(state)
    return append_log(graph_state, "已执行 realtime_quote_node")


def market_node(graph_state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    """LangGraph 市场分析节点。"""
    state = graph_state["agent_state"]
    config = graph_state["config"]
    real_workflow.run_real_market_node(state, config=config)
    return append_log(graph_state, "已执行 market_node")


def news_node(graph_state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    """LangGraph 新闻分析节点。"""
    state = graph_state["agent_state"]
    config = graph_state["config"]
    real_workflow.run_real_news_node(state, config=config)
    return append_log(graph_state, "已执行 news_node")


def summary_node(graph_state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    """LangGraph 实时综合节点。"""
    state = graph_state["agent_state"]
    config = graph_state["config"]
    real_workflow.run_real_summary_node(state, config=config)
    return append_log(graph_state, "已执行 summary_node")


def risk_node(graph_state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    """LangGraph 实时风控节点。"""
    state = graph_state["agent_state"]
    config = graph_state["config"]
    real_workflow.run_real_risk_node(state, config=config)
    return append_log(graph_state, "已执行 risk_node")


def trader_node(graph_state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    """LangGraph 实时交易员节点。"""
    state = graph_state["agent_state"]
    config = graph_state["config"]
    real_workflow.run_real_trader_node(state, config=config)
    return append_log(graph_state, "已执行 trader_node")


def build_langgraph_workflow():
    """构建 LangGraph 工作流。

    这是本文件最核心的函数。

    你可以按下面的顺序理解：

    1. StateGraph(LangGraphWorkflowState)
       创建一个图。

    2. add_node(...)
       把 Python 函数注册成图里的节点。

    3. add_conditional_edges(...)
       添加条件边。
       条件边的意思是：
       不固定写死下一步去哪，
       而是运行 route_next_node() 来决定。

    4. compile()
       把图编译成可以运行的对象。
    """
    graph = StateGraph(LangGraphWorkflowState)

    # 注册节点。
    #
    # 左边是节点名。
    # 右边是节点函数。
    #
    # 节点名必须和第 24 步路由器返回的名字一致。
    graph.add_node("realtime_quote_node", realtime_quote_node)
    graph.add_node("market_node", market_node)
    graph.add_node("news_node", news_node)
    graph.add_node("summary_node", summary_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("trader_node", trader_node)

    # path_map 告诉 LangGraph：
    #
    # route_next_node() 返回某个字符串时，
    # 应该跳转到图里的哪个节点。
    #
    # 特别注意：
    # 路由器返回 "done" 时，
    # 对应的是 LangGraph 的 END。
    # END 是 LangGraph 内置的结束节点。
    path_map = {
        "realtime_quote_node": "realtime_quote_node",
        "market_node": "market_node",
        "news_node": "news_node",
        "summary_node": "summary_node",
        "risk_node": "risk_node",
        "trader_node": "trader_node",
        "done": END,
    }

    # 从 START 开始，不直接指定固定节点，
    # 而是先调用 route_next_node() 判断。
    #
    # 这表示：
    # 图一启动，就先问路由器“当前 state 缺什么？”
    graph.add_conditional_edges(
        START,
        route_next_node,
        path_map=path_map,
    )

    # 每个节点执行完成后，也不写死下一个节点，
    # 而是再次调用 route_next_node()。
    #
    # 这就是 LangGraph 里的条件路由。
    #
    # 如果 market_node 执行后已经有 market_report，
    # 路由器下一次就会返回 news_node。
    #
    # 如果所有报告都有了，
    # 路由器就会返回 done，
    # LangGraph 就会走向 END。
    for node_name in [
        "realtime_quote_node",
        "market_node",
        "news_node",
        "summary_node",
        "risk_node",
        "trader_node",
    ]:
        graph.add_conditional_edges(
            node_name,
            route_next_node,
            path_map=path_map,
        )

    return graph.compile()


def run_langgraph_real_workflow(
    symbol: str,
    start_date: str,
    end_date: str,
    config: real_workflow.RealWorkflowConfig,
) -> LangGraphWorkflowState:
    """运行 LangGraph 真实工作流。

    这里不再自己写 for 循环。

    我们只做三件事：

    1. 创建初始状态。
    2. 构建并编译 LangGraph。
    3. 调用 app.invoke(initial_state)。

    LangGraph 会自动根据条件边推进流程。
    """
    initial_agent_state = agent_state.TradingAgentState(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=config.provider,
        model=config.model,
    )
    initial_graph_state: LangGraphWorkflowState = {
        "agent_state": initial_agent_state,
        "config": config,
        "step_log": [],
    }
    app = build_langgraph_workflow()

    # recursion_limit 是 LangGraph 的递归/步数保护。
    #
    # 如果路由规则写错，导致图一直循环，
    # LangGraph 会在超过这个限制后报错。
    #
    # 它的作用类似第 26 步里的 max_steps。
    return app.invoke(
        initial_graph_state,
        config={"recursion_limit": 20},
    )


def render_langgraph_workflow_report(
    graph_state: LangGraphWorkflowState,
) -> str:
    """把 LangGraph 工作流最终结果渲染成文本。"""
    state = graph_state["agent_state"]
    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(graph_state["step_log"], start=1)
    )

    return f"""{agent_state.render_state_summary(state)}

======== LangGraph 执行日志 ========
{step_log}

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


def demo_langgraph_real_workflow() -> None:
    """演示真正的 LangGraph 工作流。

    默认仍然使用 mock 模型。

    原因：
    当前重点是学习 LangGraph 如何推动节点运行。
    如果要真实调用 DeepSeek，
    可以设置环境变量：

    LLM_PROVIDER=deepseek
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    provider = os.environ.get("LLM_PROVIDER", "mock")
    model = os.environ.get("LLM_MODEL") or None
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "3"))

    config = real_workflow.RealWorkflowConfig(
        provider=provider,
        model=model,
        news_max_items=news_max_items,
    )
    graph_state = run_langgraph_real_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    print(render_langgraph_workflow_report(graph_state))


if __name__ == "__main__":
    demo_langgraph_real_workflow()
