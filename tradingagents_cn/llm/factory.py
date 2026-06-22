"""多模型 LLM 工厂。

这个文件解决一个问题：

    主流程不要写死 DeepSeek。

以后你给什么 API Key，就可以通过 provider/model 切换到对应模型。
目前支持：

    deepseek
    openai
    kimi
    gemini

注意：
    这里只负责创建客户端，不负责判断哪个模型更聪明。
"""

from __future__ import annotations

import os
from typing import Any

from tradingagents_cn.llm.deepseek_client import DeepSeekChatClient
from tradingagents_cn.llm.fallback_client import FallbackChatClient, build_fallback_chat_client
from tradingagents_cn.llm.openai_compatible_client import OpenAICompatibleChatClient


DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "kimi": "moonshot-v1-8k",
    "gemini": "gemini-2.5-flash",
}


DEFAULT_FALLBACK_PROVIDERS = ("deepseek", "gemini")


PROVIDER_ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "kimi": "KIMI_API_KEY",
    # Google 官方 Gemini SDK 常见变量是 GOOGLE_API_KEY；
    # 很多用户也会按服务商名称设置 GEMINI_API_KEY。
    # 这里两个都支持，先读 GEMINI_API_KEY，再读 GOOGLE_API_KEY。
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "kimi": "https://api.moonshot.cn/v1/chat/completions",
    # Gemini 提供 OpenAI 兼容入口。
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}


PROVIDER_BASE_URL_ENV_KEYS = {
    "deepseek": "DEEPSEEK_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "kimi": "KIMI_BASE_URL",
    "gemini": "GEMINI_BASE_URL",
}


def create_chat_client(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
):
    """创建聊天模型客户端。

    provider:
        deepseek / openai / kimi / gemini。
        不传时读取 TRADINGAGENTS_CN_LLM_PROVIDER；
        环境变量也没有时默认使用 fallback 链：
            deepseek -> gemini

    model:
        不传时优先读取对应环境变量：
            DEEPSEEK_MODEL
            OPENAI_MODEL
            KIMI_MODEL
            GEMINI_MODEL
        仍然没有时使用 DEFAULT_MODELS。
    """
    configured_provider = provider or os.environ.get("TRADINGAGENTS_CN_LLM_PROVIDER")
    if not configured_provider and model is None:
        return create_fallback_chat_client(**kwargs)

    actual_provider = normalize_provider(configured_provider or "deepseek")
    return create_single_chat_client(
        provider=actual_provider,
        model=model or get_model_from_env(actual_provider),
        **kwargs,
    )


def create_single_chat_client(
    provider: str,
    model: str | None = None,
    **kwargs: Any,
):
    """创建单个 provider 的聊天客户端。"""
    actual_provider = normalize_provider(provider)
    actual_model = model or get_model_from_env(actual_provider)
    if actual_provider == "deepseek":
        return DeepSeekChatClient(model=actual_model, **kwargs)

    return OpenAICompatibleChatClient(
        api_key_env=PROVIDER_ENV_KEYS[actual_provider],
        model=actual_model,
        base_url=get_base_url_from_env(actual_provider),
        **kwargs,
    )


def create_fallback_chat_client(**kwargs: Any) -> FallbackChatClient:
    """创建默认兜底模型链：DeepSeek -> Gemini。

    可以通过 TRADINGAGENTS_CN_LLM_FALLBACK_PROVIDERS 覆盖顺序，例如：
        deepseek,gemini
        gemini,deepseek

    注意：
        这里不会立刻访问网络，也不会立刻校验 API Key。
        真正调用时，如果 DeepSeek 失败，FallbackChatClient 会继续尝试 Gemini。
    """
    providers = get_fallback_providers_from_env()
    clients = [
        (
            provider,
            create_single_chat_client(provider=provider, model=get_model_from_env(provider), **kwargs),
        )
        for provider in providers
    ]
    return build_fallback_chat_client(clients)


def normalize_provider(provider: str) -> str:
    """规范化模型服务商名称。"""
    value = str(provider or "").strip().lower()
    alias_map = {
        "moonshot": "kimi",
        "moonshotai": "kimi",
        "google": "gemini",
        "google-gemini": "gemini",
        "gpt": "openai",
    }
    value = alias_map.get(value, value)

    if value not in DEFAULT_MODELS:
        supported = "、".join(sorted(DEFAULT_MODELS))
        raise ValueError(f"不支持的模型服务商：{provider}。当前支持：{supported}")
    return value


def get_model_from_env(provider: str) -> str:
    """从环境变量读取模型名，读不到就用默认模型。"""
    env_name = f"{provider.upper()}_MODEL"
    return os.environ.get(env_name, DEFAULT_MODELS[provider])


def get_base_url_from_env(provider: str) -> str:
    """从环境变量读取模型接口地址，读不到就用默认地址。

    注意：
        这里需要的是完整 chat/completions 地址。
        例如 Gemini：
        https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    """
    env_name = PROVIDER_BASE_URL_ENV_KEYS[provider]
    return os.environ.get(env_name, PROVIDER_BASE_URLS[provider])


def get_fallback_providers_from_env() -> tuple[str, ...]:
    """读取默认 fallback provider 顺序。"""
    raw = os.environ.get("TRADINGAGENTS_CN_LLM_FALLBACK_PROVIDERS", "")
    if not raw.strip():
        return DEFAULT_FALLBACK_PROVIDERS

    providers: list[str] = []
    for item in raw.split(","):
        provider = item.strip()
        if not provider:
            continue
        normalized = normalize_provider(provider)
        if normalized not in providers:
            providers.append(normalized)

    return tuple(providers) or DEFAULT_FALLBACK_PROVIDERS
