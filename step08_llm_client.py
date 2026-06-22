"""第 08 步：统一大模型客户端。

这个文件只解决一件事：

给上层 Agent 一个统一入口，不管底层用的是：
- mock：本地假模型，不花钱、不联网，方便测试。
- deepseek：DeepSeek API。
- openai：OpenAI API。
- gemini：Google Gemini 的 OpenAI 兼容接口。
- kimi：月之暗面 Kimi API。

上层 Agent 不应该关心“到底是哪家模型”。
上层 Agent 只需要做一件事：

    response = call_llm(client, prompt)

这样以后换模型时，只改配置，不改 Agent 主流程。

重要安全约定：
API Key 不写进代码。
API Key 不写进日志。
API Key 只从环境变量读取。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    """大模型客户端协议。

    Protocol 可以理解成“接口约定”：

    只要一个类有 generate(prompt: str) -> str 这个方法，
    它就可以被当成 LLMClient 使用。

    这样上层代码不需要知道具体模型是谁。
    """

    name: str

    def generate(self, prompt: str) -> str:
        """接收 Prompt，返回模型生成的文本。"""
        ...


@dataclass(frozen=True)
class ProviderConfig:
    """某一家大模型平台的配置。

    字段说明：
    - provider：平台名称，比如 deepseek、openai、gemini、kimi。
    - base_url：该平台的 OpenAI 兼容 API 地址。
    - api_key_env：从哪个环境变量读取 API Key。
    - default_model：没有手动指定模型时，默认使用哪个模型。

    注意：
    这里保存的是“环境变量名字”，不是 API Key 本身。
    """

    provider: str
    base_url: str
    api_key_env: str
    default_model: str


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    # DeepSeek 官方文档给出的 OpenAI 兼容地址是 https://api.deepseek.com
    "deepseek": ProviderConfig(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-pro",
    ),
    # OpenAI 官方 Chat Completions 地址位于 https://api.openai.com/v1/chat/completions
    "openai": ProviderConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
    ),
    # Gemini 官方提供 OpenAI 兼容入口：
    # https://generativelanguage.googleapis.com/v1beta/openai/
    "gemini": ProviderConfig(
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash",
    ),
    # Kimi 官方文档给出的 OpenAI 兼容地址是 https://api.moonshot.cn/v1
    "kimi": ProviderConfig(
        provider="kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-k2.6",
    ),
}


def read_secret_env(name: str) -> str | None:
    """读取密钥环境变量。

    优先读取当前 Python 进程里的环境变量。

    但在 Windows 上有一个容易踩坑的地方：
    如果你刚刚在“系统环境变量”里新增了 API Key，
    已经打开的 Codex / PowerShell / Python 进程可能还看不到。

    所以这里额外尝试读取 Windows 注册表里的：
    - 当前用户环境变量
    - 系统环境变量

    注意：
    这个函数只返回给程序内部使用。
    不要把返回值 print 出来。
    """
    value = os.environ.get(name)
    if value:
        return value

    if os.name != "nt":
        return None

    try:
        import winreg
    except ImportError:
        return None

    registry_locations = [
        (
            winreg.HKEY_CURRENT_USER,
            "Environment",
        ),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ]

    for root, path in registry_locations:
        try:
            with winreg.OpenKey(root, path) as key:
                registry_value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue

        if isinstance(registry_value, str) and registry_value:
            return registry_value

    return None


@dataclass
class MockLLMClient:
    """假的大模型客户端。

    它不联网、不需要 API Key、不花钱。

    为什么保留 mock：
    - 写代码时可以快速测试流程。
    - 没有 API Key 时也能学习项目结构。
    - 后面写自动化测试时，不应该每次都真的请求大模型。
    """

    name: str = "mock_llm"

    def generate(self, prompt: str) -> str:
        """根据 Prompt 中的关键词生成一份模拟报告。"""
        short_trend = "短线趋势暂时不明确"
        if "高于 MA5" in prompt:
            short_trend = "短线价格站上 MA5，技术面偏强"
        elif "低于 MA5" in prompt:
            short_trend = "短线价格跌破 MA5，技术面偏弱"

        volume_view = "量能变化不明显"
        if "成交量明显放大" in prompt or "成交量温和放大" in prompt:
            volume_view = "成交量出现放大，说明交易活跃度提高"
        elif "成交量明显缩小" in prompt:
            volume_view = "成交量明显缩小，说明资金参与度下降"

        close_view = "日内收盘位置中性"
        if "尾盘表现偏强" in prompt:
            close_view = "收盘接近日内高位，尾盘承接较好"
        elif "尾盘表现偏弱" in prompt:
            close_view = "收盘接近日内低位，尾盘抛压较明显"

        return f"""## 技术面结论
{short_trend}。{volume_view}。{close_view}。

## 关键证据
- 市场快照中包含收盘价与均线的相对位置。
- 市场快照中包含成交量相对 5 日均量的变化。
- 市场快照中包含收盘价在当日 K 线中的位置。

## 风险与分歧
- 当前报告只基于技术面快照，没有结合新闻、公告、基本面和情绪面。
- 如果 MA20 或更长周期数据不足，中期趋势判断需要谨慎。
- 技术面信号不能单独作为最终交易依据。

## 给后续 Agent 的提示
- 新闻 Agent 需要检查最近是否有公告、政策或行业催化。
- 情绪 Agent 需要检查股吧、雪球等社区是否存在一致预期或过热风险。
- 风控 Agent 需要结合波动率、支撑位和仓位管理进一步判断。"""


@dataclass
class OpenAICompatibleLLMClient:
    """OpenAI 兼容大模型客户端。

    很多大模型平台都兼容 OpenAI 的 Chat Completions 请求格式。
    也就是说，虽然平台不同，但请求大体长这样：

    POST {base_url}/chat/completions
    Authorization: Bearer {api_key}
    {
        "model": "...",
        "messages": [...]
    }

    所以 DeepSeek、OpenAI、Gemini、Kimi 可以共用这一套代码。
    """

    config: ProviderConfig
    model: str | None = None
    temperature: float = 0.2
    timeout: float = 60.0

    @property
    def name(self) -> str:
        """返回当前模型来源名称。"""
        return self.config.provider

    @property
    def model_name(self) -> str:
        """返回真正使用的模型名。"""
        return self.model or self.config.default_model

    def generate(self, prompt: str) -> str:
        """调用真实大模型，并返回模型文本。

        如果没有设置对应 API Key，这里会直接报错。
        这样可以避免程序悄悄失败，或者误以为已经调用成功。
        """
        api_key = read_secret_env(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"缺少环境变量 {self.config.api_key_env}，"
                f"无法调用 {self.config.provider}。"
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个严谨的 A 股投研分析助手。"
                        "请只基于用户提供的数据分析，不要编造不存在的数据。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "stream": False,
        }

        request = urllib.request.Request(
            url=self._chat_completions_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_text = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.config.provider} API 返回 HTTP {error.code}：{error_text}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"无法连接 {self.config.provider} API：{error}"
            ) from error

        return self._parse_chat_completion_text(raw_text)

    def _chat_completions_url(self) -> str:
        """拼出 Chat Completions 请求地址。"""
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _parse_chat_completion_text(self, raw_text: str) -> str:
        """从 OpenAI 兼容响应中取出 assistant 的文本。"""
        data = json.loads(raw_text)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"{self.config.provider} API 返回结构不符合预期：{raw_text}"
            ) from error

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"{self.config.provider} API 返回了空内容：{raw_text}"
            )
        return content


@dataclass
class LLMResponse:
    """统一的大模型返回结构。

    字段说明：
    - text：模型返回的正文。
    - provider：模型平台，比如 mock_llm、deepseek、openai。
    - model：真实模型名；mock 没有真实模型名，所以可以为空。
    """

    text: str
    provider: str
    model: str | None = None


def create_llm_client(
    provider: str = "mock",
    model: str | None = None,
    temperature: float = 0.2,
) -> LLMClient:
    """根据 provider 创建大模型客户端。

    使用方式示例：

    create_llm_client("mock")
    create_llm_client("deepseek")
    create_llm_client("openai", model="gpt-4.1-mini")
    create_llm_client("gemini", model="gemini-3.5-flash")
    create_llm_client("kimi", model="kimi-k2.6")

    如果 provider 是 mock，就返回本地假模型。
    如果 provider 是真实平台，就返回 OpenAICompatibleLLMClient。
    """
    normalized_provider = provider.strip().lower()

    if normalized_provider == "mock":
        return MockLLMClient()

    if normalized_provider not in PROVIDER_CONFIGS:
        available = ", ".join(["mock", *PROVIDER_CONFIGS.keys()])
        raise ValueError(
            f"不支持的大模型 provider：{provider}。可选值：{available}"
        )

    return OpenAICompatibleLLMClient(
        config=PROVIDER_CONFIGS[normalized_provider],
        model=model,
        temperature=temperature,
    )


def call_llm(client: LLMClient, prompt: str) -> LLMResponse:
    """通过统一接口调用大模型客户端。

    当前流程：
    1. 接收一个客户端。
    2. 把 Prompt 交给客户端。
    3. 把返回文本包装成 LLMResponse。

    上层 Agent 只调用这个函数。
    至于底层是 mock、DeepSeek、OpenAI、Gemini、Kimi，
    上层 Agent 不需要关心。
    """
    text = client.generate(prompt)
    provider = getattr(client, "name", client.__class__.__name__)
    model = getattr(client, "model_name", None)
    return LLMResponse(
        text=text,
        provider=provider,
        model=model,
    )


def demo_mock_call() -> None:
    """演示 mock 模型调用。

    这个演示不会联网，也不需要 API Key。
    """
    demo_prompt = """股票代码：600519
市场快照：
- 收盘价高于 MA5，短期价格相对强于该均线。
- 成交量明显放大，约为 5 日均量的 1.80 倍。
- 收盘接近日内高位，说明尾盘表现偏强。"""

    client = create_llm_client("mock")
    response = call_llm(client, demo_prompt)
    print(f"模型来源：{response.provider}")
    print(f"模型名称：{response.model}")
    print(response.text)


def build_demo_prompt() -> str:
    """构造一段用于测试真实大模型的演示 Prompt。

    这里故意不用真实行情，避免第 08 步依赖 AKShare。
    第 08 步只负责“大模型能不能被统一调用”。
    """
    return """股票代码：600519
市场快照：
- 收盘价高于 MA5，短期价格相对强于该均线。
- 收盘价高于 MA10，短期价格相对强于该均线。
- 收盘价低于 MA20，短期价格相对弱于该均线。
- 成交量温和放大，约为 5 日均量的 1.20 倍。
- 收盘接近日内高位，说明尾盘表现偏强。

请你用 A 股投研分析师的口吻，输出：
1. 技术面结论
2. 关键证据
3. 风险与分歧
4. 给后续新闻 Agent、情绪 Agent、风控 Agent 的提示
"""


def demo_real_llm_call(
    provider: str = "deepseek",
    model: str | None = None,
) -> None:
    """演示真实大模型调用。

    默认调用 DeepSeek。

    如果以后要换模型，有两种方式：
    1. 改函数参数，比如 demo_real_llm_call("openai")。
    2. 设置环境变量 LLM_PROVIDER，比如 LLM_PROVIDER=openai。

    注意：
    这个函数会真实请求 API。
    是否收费取决于对应平台的计费规则。
    """
    client = create_llm_client(provider=provider, model=model)
    response = call_llm(client, build_demo_prompt())

    print(f"模型来源：{response.provider}")
    print(f"模型名称：{response.model}")
    print(response.text)


def print_real_provider_config() -> None:
    """打印真实 provider 配置。

    只打印环境变量名，不打印 API Key。
    """
    for provider in ["deepseek", "openai", "gemini", "kimi"]:
        client = create_llm_client(provider)
        key_status = "已设置" if read_secret_env(client.config.api_key_env) else "未设置"
        print(
            f"{provider}: "
            f"env={client.config.api_key_env}, "
            f"key={key_status}, "
            f"base_url={client.config.base_url}, "
            f"model={client.model_name}"
        )


if __name__ == "__main__":
    # 直接运行本文件时，默认用真实 DeepSeek API。
    # 如果你以后想切换模型，可以设置：
    # LLM_PROVIDER=openai / gemini / kimi / deepseek
    # LLM_MODEL=具体模型名
    selected_provider = os.environ.get("LLM_PROVIDER", "deepseek")
    selected_model = os.environ.get("LLM_MODEL") or None

    print("可用真实模型配置：")
    print_real_provider_config()
    print("\n开始调用真实大模型：")
    demo_real_llm_call(provider=selected_provider, model=selected_model)
