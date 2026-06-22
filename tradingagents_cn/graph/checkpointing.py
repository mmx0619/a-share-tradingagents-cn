"""LangGraph checkpoint 辅助函数。

checkpoint 的作用：

    让一次图执行拥有 thread_id。
    如果中途失败或后续要继续同一个会话，可以用同一个 thread_id 恢复上下文。

当前正式工程默认使用 SQLite checkpoint。
它会把图状态写入本地数据库文件，程序退出后仍然保留。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_SQLITE_CHECKPOINT_PATH = "outputs/checkpoints/tradingagents_cn_checkpoints.sqlite"


@dataclass(frozen=True)
class CheckpointThreadInfo:
    """SQLite checkpoint 中一个 thread 的摘要信息。

    thread_id:
        LangGraph 运行时使用的线程编号。

    checkpoint_ns:
        LangGraph checkpoint namespace。
        当前主链路一般为空字符串，但保留这个字段可以兼容以后更复杂的图。

    checkpoint_count:
        这个 thread 已经写入了多少个 checkpoint。

    latest_checkpoint_id:
        最新 checkpoint 的 id。
        resume 时不是手动传这个 id，而是继续使用 thread_id。

    latest_step:
        metadata 里的 step。
        它可以帮助判断图大概跑到了第几步。

    latest_source:
        metadata 里的 source，例如 input / loop。
    """

    thread_id: str
    checkpoint_ns: str
    checkpoint_count: int
    latest_checkpoint_id: str
    latest_step: int | None
    latest_source: str


def create_memory_checkpointer() -> MemorySaver:
    """创建内存版 checkpointer。"""
    return MemorySaver()


def get_default_sqlite_checkpoint_path() -> Path:
    """返回默认 SQLite checkpoint 数据库路径。

    可以用环境变量覆盖：

        TRADINGAGENTS_CN_CHECKPOINT_DB
    """
    return Path(
        os.environ.get(
            "TRADINGAGENTS_CN_CHECKPOINT_DB",
            DEFAULT_SQLITE_CHECKPOINT_PATH,
        )
    )


@contextmanager
def create_sqlite_checkpointer(
    db_path: str | Path | None = None,
) -> Iterator[SqliteSaver]:
    """创建 SQLite 版 checkpointer。

    注意：
        SqliteSaver.from_conn_string(...) 是 context manager。
        所以调用方必须在 with 代码块里编译和运行 LangGraph。
    """
    actual_path = Path(db_path) if db_path is not None else get_default_sqlite_checkpoint_path()
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(actual_path)) as checkpointer:
        yield checkpointer


def build_thread_config(thread_id: str) -> dict:
    """构造 LangGraph invoke/stream 使用的 thread 配置。"""
    return {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }


def checkpoint_database_exists(db_path: str | Path | None = None) -> bool:
    """判断 SQLite checkpoint 数据库是否已经存在。"""
    actual_path = Path(db_path) if db_path is not None else get_default_sqlite_checkpoint_path()
    return actual_path.exists()


def list_checkpoint_thread_ids(db_path: str | Path | None = None) -> list[str]:
    """列出 checkpoint 数据库里已有的 thread_id。

    这个函数直接读取 SQLite 表。
    如果数据库还不存在，返回空列表。
    """
    actual_path = Path(db_path) if db_path is not None else get_default_sqlite_checkpoint_path()
    if not actual_path.exists():
        return []

    try:
        with sqlite3.connect(actual_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
    except sqlite3.Error:
        return []

    return [str(row[0]) for row in rows if row and row[0] is not None]


def list_checkpoint_threads(
    db_path: str | Path | None = None,
) -> list[CheckpointThreadInfo]:
    """列出 checkpoint 数据库里已有 thread 的详细摘要。

    这个函数只读取 SQLite 的公开表字段：
        thread_id
        checkpoint_ns
        checkpoint_id
        metadata

    不解析 checkpoint BLOB。
    原因是当前环境没有 msgpack 依赖，
    而 thread 列表只需要展示“是否存在、最新 step、checkpoint 数量”。
    """
    actual_path = Path(db_path) if db_path is not None else get_default_sqlite_checkpoint_path()
    if not actual_path.exists():
        return []

    try:
        with sqlite3.connect(actual_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    latest.thread_id,
                    latest.checkpoint_ns,
                    latest.latest_checkpoint_id,
                    latest.checkpoint_count,
                    c.metadata
                FROM (
                    SELECT
                        thread_id,
                        checkpoint_ns,
                        MAX(checkpoint_id) AS latest_checkpoint_id,
                        COUNT(*) AS checkpoint_count
                    FROM checkpoints
                    GROUP BY thread_id, checkpoint_ns
                ) AS latest
                JOIN checkpoints AS c
                  ON c.thread_id = latest.thread_id
                 AND c.checkpoint_ns = latest.checkpoint_ns
                 AND c.checkpoint_id = latest.latest_checkpoint_id
                ORDER BY latest.thread_id, latest.checkpoint_ns
                """
            ).fetchall()
    except sqlite3.Error:
        return []

    infos: list[CheckpointThreadInfo] = []
    for row in rows:
        metadata = parse_checkpoint_metadata(row[4])
        infos.append(
            CheckpointThreadInfo(
                thread_id=str(row[0]),
                checkpoint_ns=str(row[1] or ""),
                latest_checkpoint_id=str(row[2]),
                checkpoint_count=int(row[3] or 0),
                latest_step=parse_optional_int(metadata.get("step")),
                latest_source=str(metadata.get("source") or "unknown"),
            )
        )
    return infos


def parse_checkpoint_metadata(raw_metadata: object) -> dict:
    """解析 checkpoint metadata JSON。"""
    if raw_metadata is None:
        return {}

    if isinstance(raw_metadata, bytes):
        text = raw_metadata.decode("utf-8", errors="ignore")
    else:
        text = str(raw_metadata)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def parse_optional_int(value: object) -> int | None:
    """把 metadata step 转成 int，失败时返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def has_checkpoint_for_thread(thread_id: str, db_path: str | Path | None = None) -> bool:
    """判断某个 thread_id 是否已有 checkpoint。"""
    return str(thread_id) in set(list_checkpoint_thread_ids(db_path))
