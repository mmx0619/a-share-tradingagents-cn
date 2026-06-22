"""命令行进度提示工具。

完整 TradingAgents 链路会调用多个数据源和大模型。
如果终端长时间没有输出，用户很容易以为程序卡死。
这个模块只负责打印“现在跑到哪一步”，不参与任何投资判断。
"""

from __future__ import annotations

from datetime import datetime


def emit_progress(message: str, enabled: bool = True) -> None:
    """输出一行进度提示。

    enabled:
        测试或嵌入式调用可以关闭进度输出。
    """
    if not enabled:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
