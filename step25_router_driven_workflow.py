"""第 25 步：路由驱动的工作流执行器。

第 24 步已经完成了路由器：

当前 state
  ↓
decide_next_node(state)
  ↓
得到下一步节点

但是第 24 步只负责“判断”，还没有真正“执行”。

当前文件做第 25 件事：

创建 state
  ↓
问路由器：下一步去哪？
  ↓
执行对应节点
  ↓
再问路由器：下一步去哪？
  ↓
一直循环到 done

这一步就是 LangGraph 的雏形：

- state：共享状态
- router：决定下一步节点
- node：真正执行的函数
- loop：不断推进流程

注意：
为了让这个文件跑得快、好理解，
这里先用“模拟节点”。

也就是说：
不联网，不调用 DeepSeek，不获取真实行情。

目的不是再次验证数据源，
而是让你看懂“路由如何推动流程前进”。
"""

from __future__ import annotations

from collections.abc import Callable

import step17_agent_state as agent_state
import step24_agent_router as router


# NodeHandler 是一个“类型说明”，不是一个真正执行的函数。
#
# 它的意思是：
#
# 以后凡是可以作为“节点函数”的东西，都应该长这样：
#
#   输入：TradingAgentState
#   输出：None
#
# 为什么输出是 None？
#
# 因为节点函数不需要返回一个新的 state。
# 它会直接修改传进来的 state。
#
# 例如：
#
#   state.market_report = "市场报告"
#
# 这就表示这个节点已经把自己的工作结果写入共享状态了。
NodeHandler = Callable[[agent_state.TradingAgentState], None]


def mock_realtime_quote_node(state: agent_state.TradingAgentState) -> None:
    """模拟实时行情节点。

    真实版本会调用 step19_realtime_quote.py。
    这里为了演示路由，只写入一段模拟文本。

    你可以把这个函数理解成：

    - 真实系统里，它负责获取实时行情。
    - 当前教学版里，它不联网，只是伪造一段文本。
    - 它完成后，会把结果写到 state.realtime_quote_text。

    只要 state.realtime_quote_text 有值，
    第 24 步的路由器下次就不会再让流程回到 realtime_quote_node。
    """
    state.realtime_quote_text = "模拟实时行情快照。"


def mock_market_node(state: agent_state.TradingAgentState) -> None:
    """模拟市场分析师节点。

    真实版本会做这些事：

    1. 获取历史行情。
    2. 计算均线、涨跌幅、成交量等技术指标。
    3. 生成市场快照。
    4. 把市场快照放进 Prompt。
    5. 调用大模型生成市场技术面报告。

    这里为了让你先看懂“路由执行”，
    暂时不做上面这些复杂动作，
    只是假装已经生成了市场快照和市场报告。
    """
    # market_snapshot_text 是“市场快照文本”。
    # 它通常来自前面的行情和技术指标计算。
    state.market_snapshot_text = "模拟日线市场快照。"

    # set_market_report() 不只是写入 market_report。
    # 它还会顺便记录一条 AgentMessage，
    # 表示 market_agent 已经运行过一次。
    state.set_market_report(
        report="模拟市场技术面报告。",
        provider="mock",
        model="mock",
    )


def mock_news_node(state: agent_state.TradingAgentState) -> None:
    """模拟新闻分析节点。

    真实版本会做这些事：

    1. 获取东方财富等来源的个股新闻。
    2. 用规则提取新闻事件。
    3. 把新闻事件放进 Prompt。
    4. 调用大模型生成新闻面报告。

    当前教学版只写入模拟文本。
    """
    # news_events_text 表示已经整理好的新闻事件。
    # 后面的新闻 Agent 会基于这些事件生成新闻面报告。
    state.news_events_text = "模拟新闻事件信号。"

    # 写入新闻面报告。
    # 只要 state.news_report 有值，
    # 路由器下次就会认为新闻节点已经完成。
    state.set_news_report(
        report="模拟新闻面报告。",
        provider="mock",
        model="mock",
    )


def mock_summary_node(state: agent_state.TradingAgentState) -> None:
    """模拟综合汇总节点。

    真实版本会读取：

    - state.market_report
    - state.news_report

    然后把这两个报告合在一个 Prompt 里，
    让综合汇总 Agent 判断：

    - 技术面和新闻面是否互相支持？
    - 有没有冲突？
    - 当前更偏机会还是风险？

    当前教学版只写入一段模拟综合报告。
    """
    state.set_summary_report(
        report="模拟综合汇总报告。",
        provider="mock",
        model="mock",
    )


def mock_risk_node(state: agent_state.TradingAgentState) -> None:
    """模拟风控节点。

    真实版本会读取 state.summary_report。

    风控 Agent 的核心任务不是“找买点”，
    而是判断：

    - 能不能交易？
    - 最大风险在哪里？
    - 有哪些动作必须禁止？
    - 如果要观察，需要等哪些条件？

    当前教学版只写入一段模拟风控报告。
    """
    state.set_risk_report(
        report="模拟风控报告。",
        provider="mock",
        model="mock",
    )


def mock_trader_node(state: agent_state.TradingAgentState) -> None:
    """模拟交易员节点。

    真实版本会读取 state.risk_report。

    交易员 Agent 的任务是：

    - 在风控允许的范围内制定交易预案。
    - 如果风控禁止交易，交易员不能强行给买入建议。
    - 输出应该是“条件式预案”，而不是无条件喊买喊卖。

    当前教学版只写入一段模拟交易员预案。
    """
    state.set_trader_plan(
        plan="模拟交易员预案。",
        provider="mock",
        model="mock",
    )


# 这是第 25 步最关键的结构之一。
#
# 它把“节点名称字符串”和“真正的 Python 函数”对应起来。
#
# 例如路由器返回：
#
#   decision.next_node == "market_node"
#
# 那么执行器就可以执行：
#
#   handler = MOCK_NODE_HANDLERS["market_node"]
#   handler(state)
#
# 等价于：
#
#   mock_market_node(state)
#
# 这样，路由器只需要返回一个固定名字，
# 执行器就能找到对应函数并运行。
MOCK_NODE_HANDLERS: dict[str, NodeHandler] = {
    "realtime_quote_node": mock_realtime_quote_node,
    "market_node": mock_market_node,
    "news_node": mock_news_node,
    "summary_node": mock_summary_node,
    "risk_node": mock_risk_node,
    "trader_node": mock_trader_node,
}


def run_router_driven_workflow(
    state: agent_state.TradingAgentState,
    node_handlers: dict[str, NodeHandler],
    max_steps: int = 20,
) -> agent_state.TradingAgentState:
    """运行路由驱动的工作流。

    参数说明：
    - state：共享状态。
    - node_handlers：节点名称到函数的映射。
    - max_steps：最多执行多少步，防止路由写错后死循环。

    核心逻辑：
    1. 路由器根据 state 判断下一节点。
    2. 如果下一节点是 done，流程结束。
    3. 如果不是 done，就找到对应节点函数并执行。
    4. 节点执行后会更新 state。
    5. 回到第 1 步。
    """
    # for 循环表示最多尝试执行 max_steps 次。
    #
    # 为什么需要 max_steps？
    #
    # 如果路由规则写错了，可能出现这种情况：
    #
    #   路由器一直返回 market_node
    #   market_node 又一直没有正确写入 state.market_report
    #   于是流程永远结束不了
    #
    # max_steps 就是一个保险丝。
    # 超过指定步数还没结束，就主动报错，避免程序死循环。
    for step_index in range(1, max_steps + 1):
        # 第一步：把当前 state 交给路由器。
        #
        # 路由器会检查 state 里面缺什么。
        # 缺实时行情，就返回 realtime_quote_node。
        # 缺市场报告，就返回 market_node。
        # 全部都有了，就返回 done。
        decision = router.decide_next_node(state)
        print(f"第 {step_index} 步路由结果：{decision.next_node}")
        print(f"原因：{decision.reason}")

        # 如果路由器返回 done，
        # 说明所有主要 Agent 的结果都已经写进 state。
        # 这时流程结束，直接把最终 state 返回。
        if decision.next_node == "done":
            print("流程结束。")
            return state

        # 如果不是 done，
        # decision.next_node 就会是某个节点名，例如：
        #
        #   "market_node"
        #
        # 这里用这个节点名去 node_handlers 字典里找真正的函数。
        handler = node_handlers.get(decision.next_node)

        # 如果找不到，说明出现了代码配置错误。
        #
        # 例如：
        # 路由器返回了 "abc_node"，
        # 但 node_handlers 字典里没有 "abc_node"。
        #
        # 这种情况必须立刻报错，
        # 否则程序不知道该执行哪个节点。
        if handler is None:
            raise ValueError(f"没有找到节点处理函数：{decision.next_node}")

        # 找到函数以后，真正执行这个节点。
        #
        # 例如：
        #
        #   decision.next_node 是 "market_node"
        #   handler 就是 mock_market_node
        #
        # 那这一行就等价于：
        #
        #   mock_market_node(state)
        #
        # 节点函数会修改 state。
        # 修改完成后，下一轮循环会再次调用路由器。
        handler(state)
        print(f"已执行节点：{decision.next_node}")
        print()

    # 如果 for 循环跑完了还没有 return，
    # 说明流程在 max_steps 次以内没有走到 done。
    # 这通常意味着路由规则或节点写 state 的逻辑有问题。
    raise RuntimeError(f"超过最大执行步数 {max_steps}，可能发生了路由死循环。")


def demo_router_driven_workflow() -> None:
    """演示路由如何推动整个流程自动前进。

    这个 demo 的重点不是分析股票，
    而是观察流程如何自动从一个节点走到下一个节点。
    """
    # 创建一个空的共享状态。
    #
    # 刚创建时：
    # - 没有实时行情
    # - 没有市场报告
    # - 没有新闻报告
    # - 没有综合报告
    # - 没有风控报告
    # - 没有交易预案
    #
    # 所以路由器第一次一定会返回 realtime_quote_node。
    state = agent_state.TradingAgentState(
        symbol="002361",
        start_date="2026-01-01",
        end_date="2026-06-15",
    )

    # 把空 state 交给工作流执行器。
    #
    # 执行器会反复做：
    #
    #   问路由器下一步去哪
    #   执行对应节点
    #   节点写入 state
    #   再问路由器下一步去哪
    #
    # 直到路由器返回 done。
    final_state = run_router_driven_workflow(
        state=state,
        node_handlers=MOCK_NODE_HANDLERS,
    )

    # 打印最终 state 摘要，方便确认每个节点都已经执行过。
    print(agent_state.render_state_summary(final_state))


if __name__ == "__main__":
    demo_router_driven_workflow()
