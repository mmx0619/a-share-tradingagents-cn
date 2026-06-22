import tomllib
import unittest
from pathlib import Path

import tradingagents_cn
from tradingagents_cn import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectPackagingTest(unittest.TestCase):
    def test_pyproject_should_define_package_and_cli_entry(self):
        """pyproject.toml 应声明包名、依赖和 CLI 入口。"""
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual("tradingagents-cn", data["project"]["name"])
        self.assertEqual("run_user_question:main", data["project"]["scripts"]["tradingagents-cn"])
        self.assertIn("langgraph-checkpoint-sqlite", data["project"]["dependencies"])
        self.assertIn("akshare", data["project"]["dependencies"])

    def test_env_example_should_include_required_keys(self):
        """.env.example 应列出主要环境变量，但不包含真实密钥。"""
        text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertIn("TRADINGAGENTS_CN_LLM_PROVIDER", text)
        self.assertIn("TRADINGAGENTS_CN_CHECKPOINT_DB", text)
        self.assertIn("TRADINGAGENTS_CN_CACHE_DIR", text)
        self.assertIn("替换成你自己的key", text)

    def test_root_readme_should_point_to_user_manual(self):
        """根目录 README 应指向使用手册和开发日志。"""
        text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("使用手册与断点续跑说明.md", text)
        self.assertIn("项目目标与开发日志.md", text)
        self.assertIn("tradingagents-cn --question", text)

    def test_package_should_expose_version_and_cli_main(self):
        """包应有版本号，包内 CLI 应复用 main。"""
        self.assertEqual("0.1.0", tradingagents_cn.__version__)
        self.assertTrue(callable(cli.main))


if __name__ == "__main__":
    unittest.main()
