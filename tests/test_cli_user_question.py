import tempfile
import unittest
from pathlib import Path

import run_user_question as cli
from tradingagents_cn.llm.errors import LLMAPIError


class UserQuestionCLITest(unittest.TestCase):
    def test_parse_args_should_accept_question_and_runtime_options(self):
        """CLI 应支持直接传问题、日期、模型、轮数等参数。"""
        args = cli.parse_args(
            [
                "--question",
                "京东方A能不能买",
                "--date",
                "2026-06-18",
                "--provider",
                "deepseek",
                "--model",
                "deepseek-chat",
                "--debate-rounds",
                "2",
                "--risk-rounds",
                "3",
                "--no-deep-screening",
                "--rule-router",
            ]
        )

        self.assertEqual("京东方A能不能买", args.question)
        self.assertEqual("2026-06-18", args.date)
        self.assertEqual("deepseek", args.provider)
        self.assertEqual("deepseek-chat", args.model)
        self.assertEqual(2, args.debate_rounds)
        self.assertEqual(3, args.risk_rounds)
        self.assertTrue(args.no_deep_screening)
        self.assertTrue(args.rule_router)

    def test_build_config_from_args_should_map_switches(self):
        """CLI 参数应正确转换成 ResearchInputConfig。"""
        args = cli.parse_args(
            [
                "--history-days",
                "300",
                "--news-max-items",
                "4",
                "--fundamentals-max-rows",
                "3",
                "--fundamentals-max-columns",
                "18",
                "--benchmark-symbol",
                "000905",
                "--benchmark-name",
                "中证500",
                "--memory-holding-days",
                "10",
                "--resolve-all-pending-memory",
                "--paper-trading",
                "--paper-ledger",
                "outputs/paper_trading/test_account.json",
                "--paper-initial-cash",
                "200000",
                "--paper-max-position-pct",
                "0.15",
                "--paper-min-trade-amount",
                "2000",
                "--paper-review-days",
                "7",
                "--no-paper-review",
                "--analysts",
                "market,sentiment",
                "--sentiment-max-items",
                "6",
                "--sentiment-sources",
                "eastmoney,xueqiu",
                "--no-realtime",
                "--no-news",
                "--no-sentiment",
                "--no-fundamentals",
                "--no-full-state",
                "--full-state-dir",
                "outputs/state_test",
                "--data-vendor",
                "market_data=akshare",
                "--tool-vendor",
                "sentiment_sources=eastmoney,xueqiu",
            ]
        )

        config = cli.build_config_from_args(args)

        self.assertEqual(300, config.history_calendar_days)
        self.assertEqual(4, config.news_max_items)
        self.assertEqual(6, config.sentiment_max_items)
        self.assertEqual(3, config.fundamentals_max_rows)
        self.assertEqual(18, config.fundamentals_max_columns)
        self.assertFalse(config.include_realtime)
        self.assertFalse(config.include_news)
        self.assertFalse(config.include_sentiment)
        self.assertFalse(config.include_fundamentals)
        self.assertEqual(("market", "sentiment"), config.selected_analysts)
        self.assertEqual("eastmoney,xueqiu", config.sentiment_sources)
        self.assertFalse(config.save_full_state)
        self.assertEqual("outputs/state_test", config.full_state_output_dir)
        self.assertEqual("000905", config.benchmark_symbol)
        self.assertEqual("中证500", config.benchmark_name)
        self.assertEqual(10, config.memory_holding_days)
        self.assertTrue(config.resolve_all_pending_memory)
        self.assertTrue(config.enable_paper_trading)
        self.assertEqual("outputs/paper_trading/test_account.json", config.paper_trading_ledger_path)
        self.assertEqual(200000, config.paper_trading_initial_cash)
        self.assertEqual(0.15, config.paper_trading_max_single_position_pct)
        self.assertEqual(2000, config.paper_trading_min_trade_amount)
        self.assertEqual(7, config.paper_trading_review_holding_days)
        self.assertFalse(config.paper_trading_review_pending)
        self.assertEqual("akshare", config.data_vendors["market_data"])
        self.assertEqual("eastmoney,xueqiu", config.tool_vendors["sentiment_sources"])

    def test_build_llm_client_from_args_should_return_none_without_model_options(self):
        """不传 provider/model 时，CLI 不提前创建模型客户端。"""
        args = cli.parse_args([])

        self.assertIsNone(cli.build_llm_client_from_args(args))

    def test_render_checkpoint_thread_list(self):
        """checkpoint thread 列表应可读。"""
        text = cli.render_checkpoint_thread_list(["thread-a", "thread-b"])

        self.assertIn("SQLite checkpoint thread 列表", text)
        self.assertIn("thread-a", text)
        self.assertIn("thread-b", text)
        self.assertEqual("暂无 SQLite checkpoint thread。", cli.render_checkpoint_thread_list([]))

    def test_render_checkpoint_thread_info_list(self):
        """checkpoint 详细列表应包含续跑所需信息。"""
        text = cli.render_checkpoint_thread_info_list(
            [
                cli.CheckpointThreadInfo(
                    thread_id="single-stock-000725-2026-06-18",
                    checkpoint_ns="",
                    checkpoint_count=12,
                    latest_checkpoint_id="checkpoint-1",
                    latest_step=8,
                    latest_source="loop",
                )
            ]
        )

        self.assertIn("thread_id：single-stock-000725-2026-06-18", text)
        self.assertIn("checkpoint 数量：12", text)
        self.assertIn("最新 step：8", text)
        self.assertIn("--resume --thread-id single-stock-000725-2026-06-18", text)
        self.assertEqual("暂无 SQLite checkpoint thread。", cli.render_checkpoint_thread_info_list([]))

    def test_list_and_render_report_paths(self):
        """CLI 应能列出最终报告文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            old_report = directory / "平安银行_000001_2026-06-17_final_report.md"
            new_report = directory / "京东方A_000725_2026-06-18_final_report.md"
            other_file = directory / "notes.txt"
            old_report.write_text("old", encoding="utf-8")
            new_report.write_text("new", encoding="utf-8")
            other_file.write_text("ignore", encoding="utf-8")

            reports = cli.list_report_paths(directory)
            rendered = cli.render_report_list(reports)

        self.assertEqual(2, len(reports))
        self.assertTrue(all(path.name.endswith("_final_report.md") for path in reports))
        self.assertIn("最终报告列表", rendered)
        self.assertIn("京东方A_000725_2026-06-18_final_report.md", rendered)

    def test_render_report_list_should_handle_empty(self):
        """没有报告时应给出明确提示。"""
        self.assertEqual("暂无最终报告文件。", cli.render_report_list([]))

    def test_render_runtime_error_should_explain_llm_api_error(self):
        """模型 API 错误应渲染成中文处理建议。"""
        text = cli.render_runtime_error(LLMAPIError("DeepSeek API 调用失败：HTTP 402"))

        self.assertIn("模型调用失败", text)
        self.assertIn("HTTP 402", text)
        self.assertIn("账户余额", text)


if __name__ == "__main__":
    unittest.main()
