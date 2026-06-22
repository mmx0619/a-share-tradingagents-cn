"""包内 CLI 入口。

这个文件让项目支持：

    python -m tradingagents_cn.cli

正式命令行逻辑仍然复用根目录的 run_user_question.py，
这样不会维护两套入口代码。
"""

from __future__ import annotations

from run_user_question import main


if __name__ == "__main__":
    main()
