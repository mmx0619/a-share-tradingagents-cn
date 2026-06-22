import tempfile
import unittest
import gc
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from tradingagents_cn.graph.checkpointing import (
    build_thread_config,
    create_sqlite_checkpointer,
    has_checkpoint_for_thread,
    list_checkpoint_thread_ids,
    list_checkpoint_threads,
)
from tradingagents_cn.graph.research_report_state_graph import should_resume_thread


class TinyState(TypedDict, total=False):
    """测试用极小图状态。"""

    value: int
    finished: bool


def add_one_node(state: TinyState) -> TinyState:
    """测试节点：给 value 加 1。"""
    return {
        **state,
        "value": state.get("value", 0) + 1,
    }


def finish_node(state: TinyState) -> TinyState:
    """测试节点：标记完成。"""
    return {
        **state,
        "finished": True,
    }


def build_tiny_app(checkpointer):
    """构造测试用 LangGraph。"""
    graph = StateGraph(TinyState)
    graph.add_node("add_one", add_one_node)
    graph.add_node("finish", finish_node)
    graph.set_entry_point("add_one")
    graph.add_edge("add_one", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


class SQLiteCheckpointingTest(unittest.TestCase):
    def test_sqlite_checkpoint_should_persist_thread_id(self):
        """SQLite checkpoint 应把 thread_id 写入本地数据库。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "checkpoint.sqlite"
            thread_id = "unit-test-thread"

            with create_sqlite_checkpointer(db_path) as checkpointer:
                app = build_tiny_app(checkpointer)
                output = app.invoke(
                    {"value": 1},
                    config=build_thread_config(thread_id),
                )
                del app
                del checkpointer
            gc.collect()

            self.assertEqual(2, output["value"])
            self.assertTrue(output["finished"])
            self.assertTrue(db_path.exists())
            self.assertIn(thread_id, list_checkpoint_thread_ids(db_path))
            self.assertTrue(has_checkpoint_for_thread(thread_id, db_path))

            infos = list_checkpoint_threads(db_path)
            self.assertEqual(1, len(infos))
            self.assertEqual(thread_id, infos[0].thread_id)
            self.assertEqual("", infos[0].checkpoint_ns)
            self.assertGreaterEqual(infos[0].checkpoint_count, 1)
            self.assertGreaterEqual(infos[0].latest_step or 0, 0)
            self.assertIn(infos[0].latest_source, {"input", "loop"})
            self.assertTrue(infos[0].latest_checkpoint_id)

    def test_sqlite_checkpoint_should_resume_with_none_input(self):
        """同一个 thread_id 可以用 None 输入读取并继续已有状态。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "checkpoint.sqlite"
            thread_id = "resume-thread"

            with create_sqlite_checkpointer(db_path) as checkpointer:
                app = build_tiny_app(checkpointer)
                first_output = app.invoke(
                    {"value": 5},
                    config=build_thread_config(thread_id),
                )
                second_output = app.invoke(
                    None,
                    config=build_thread_config(thread_id),
                )
                del app
                del checkpointer
            gc.collect()

            self.assertEqual(first_output, second_output)

    def test_should_resume_thread_should_require_resume_flag_and_existing_checkpoint(self):
        """只有 resume=True 且 thread_id 已存在时，主图才应该恢复。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "checkpoint.sqlite"
            thread_id = "resume-flag-thread"

            self.assertFalse(should_resume_thread(thread_id, True, str(db_path)))

            with create_sqlite_checkpointer(db_path) as checkpointer:
                app = build_tiny_app(checkpointer)
                app.invoke({"value": 1}, config=build_thread_config(thread_id))
                del app
                del checkpointer
            gc.collect()

            self.assertFalse(should_resume_thread(thread_id, False, str(db_path)))
            self.assertTrue(should_resume_thread(thread_id, True, str(db_path)))


if __name__ == "__main__":
    unittest.main()
