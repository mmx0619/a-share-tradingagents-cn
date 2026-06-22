"""本地文本缓存测试。

这些测试不访问网络。

我们用一个假的 fetcher 模拟“联网获取数据”，
然后检查缓存层是否按预期工作：

1. 第一次没有缓存，必须调用 fetcher。
2. 第二次缓存未过期，必须直接读本地文件。
3. force_refresh=True 时，必须忽略缓存并重新调用 fetcher。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tradingagents_cn.cache.text_cache import (
    build_cache_path,
    get_or_refresh_text_cache,
    read_cache_if_fresh,
    write_text_cache,
)


class TextCacheTest(unittest.TestCase):
    """测试通用文本缓存工具。"""

    def test_second_call_should_use_cache(self) -> None:
        """第二次调用同一个缓存键时，不应该再次调用 fetcher。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = {"count": 0}

            def fetcher() -> str:
                calls["count"] += 1
                return f"联网数据第 {calls['count']} 次"

            first = get_or_refresh_text_cache(
                cache_group="fundamentals",
                cache_key="000725:balance_sheet",
                fetcher=fetcher,
                max_age_days=30,
                cache_root=temp_dir,
            )
            second = get_or_refresh_text_cache(
                cache_group="fundamentals",
                cache_key="000725:balance_sheet",
                fetcher=fetcher,
                max_age_days=30,
                cache_root=temp_dir,
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.text, "联网数据第 1 次")
            self.assertEqual(second.text, "联网数据第 1 次")
            self.assertEqual(calls["count"], 1)

    def test_force_refresh_should_ignore_existing_cache(self) -> None:
        """force_refresh=True 时，即使已有缓存，也要重新获取。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = {"count": 0}

            def fetcher() -> str:
                calls["count"] += 1
                return f"刷新数据第 {calls['count']} 次"

            get_or_refresh_text_cache(
                cache_group="fundamentals",
                cache_key="000725:income_statement",
                fetcher=fetcher,
                max_age_days=30,
                cache_root=temp_dir,
            )
            refreshed = get_or_refresh_text_cache(
                cache_group="fundamentals",
                cache_key="000725:income_statement",
                fetcher=fetcher,
                max_age_days=30,
                force_refresh=True,
                cache_root=temp_dir,
            )

            self.assertFalse(refreshed.cache_hit)
            self.assertEqual(refreshed.text, "刷新数据第 2 次")
            self.assertEqual(calls["count"], 2)

    def test_expired_cache_should_return_none(self) -> None:
        """max_age_days=0 时，已有缓存会被视为过期。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = build_cache_path(
                cache_group="fundamentals",
                cache_key="000725:cashflow",
                cache_root=temp_dir,
            )
            write_text_cache(path, "旧缓存")

            self.assertIsNone(read_cache_if_fresh(Path(path), max_age_days=0))


if __name__ == "__main__":
    unittest.main()
