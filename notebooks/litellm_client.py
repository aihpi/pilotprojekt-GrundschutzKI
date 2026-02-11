"""LiteLLM helpers for notebooks.

Keeps config in one place and reads from environment/.env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()
except Exception:
    # Optional dependency: fall back to plain environment variables.
    pass


@dataclass(frozen=True)
class LLMConfig:
    api_base: Optional[str]
    api_key: Optional[str]
    model: str
    embedding_model: str


@dataclass(frozen=True)
class VectorDBConfig:
    provider: str
    url: Optional[str]
    api_key: Optional[str]
    collection: Optional[str]


_MODEL_COSTS_REGISTERED = False


def _require(value: Optional[str], name: str) -> str:
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_llm_config() -> LLMConfig:
    model = os.getenv("LLM_MODEL") or _require(os.getenv("GPT_OSS_120B_MODEL"), "GPT_OSS_120B_MODEL")
    embedding_model = _require(os.getenv("EMBEDDING_MODEL"), "EMBEDDING_MODEL")
    return LLMConfig(
        api_base=os.getenv("LITELLM_API_BASE"),
        api_key=os.getenv("LITELLM_API_KEY"),
        model=model,
        embedding_model=embedding_model,
    )


def _ensure_litellm_model_costs(config: Optional[LLMConfig] = None) -> None:
    """Register local model pricing to avoid LiteLLM cost-map errors for custom models."""
    global _MODEL_COSTS_REGISTERED
    if _MODEL_COSTS_REGISTERED:
        return

    try:
        import litellm
    except Exception:
        return

    cfg = config or load_llm_config()
    model_specs = {
        cfg.model: "chat",
        cfg.embedding_model: "embedding",
    }

    model_costs: dict[str, dict[str, object]] = {}

    for raw_name, mode in model_specs.items():
        if not raw_name:
            continue
        # Normalize to include provider-prefixed and bare model names.
        name = raw_name.strip()
        if "/" in name:
            provider = name.split("/", 1)[0]
        else:
            provider = "openai"

        bare = name.split("/", 1)[-1]
        candidate_keys = {name, bare, f"{provider}/{bare}"}
        for key in candidate_keys:
            if key in litellm.model_cost:
                continue
            model_costs[key] = {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "litellm_provider": provider,
                "mode": mode,
            }

    if model_costs:
        try:
            litellm.register_model(model_costs)
        except Exception:
            for key, value in model_costs.items():
                litellm.model_cost.setdefault(key, {}).update(value)
            try:
                from litellm.utils import _invalidate_model_cost_lowercase_map

                _invalidate_model_cost_lowercase_map()
            except Exception:
                pass

    _MODEL_COSTS_REGISTERED = True


def load_vectordb_config() -> VectorDBConfig:
    provider = os.getenv("VECTORDB_PROVIDER", "")
    return VectorDBConfig(
        provider=provider,
        url=os.getenv("VECTORDB_URL") or os.getenv("QDRANT_URL"),
        api_key=os.getenv("VECTORDB_API_KEY") or os.getenv("QDRANT_API_KEY"),
        collection=os.getenv("VECTORDB_COLLECTION") or os.getenv("QDRANT_COLLECTION"),
    )


def get_embeddings(
    texts: Iterable[str],
    config: Optional[LLMConfig] = None,
    batch_size: int = 32,
) -> List[List[float]]:
    from litellm import embedding

    cfg = config or load_llm_config()
    _ensure_litellm_model_costs(cfg)
    all_texts = list(texts)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    embeddings: List[List[float]] = []
    for start in range(0, len(all_texts), batch_size):
        print(f"Processing embeddings {start} to {min(start + batch_size, len(all_texts))} / {len(all_texts)}")
        batch = all_texts[start : start + batch_size]
        response = embedding(
            model=cfg.embedding_model,
            input=batch,
            encoding_format="float",
            api_key=cfg.api_key,
            api_base=cfg.api_base,
        )
        embeddings.extend([item["embedding"] for item in response["data"]])
    return embeddings


def chat_completion(messages: List[dict[str, Any]], config: Optional[LLMConfig] = None, **kwargs: Any) -> Any:
    from litellm import completion

    cfg = config or load_llm_config()
    _ensure_litellm_model_costs(cfg)
    print(f"Using model: {cfg.model}")
    return completion(
        model=cfg.model,
        messages=messages,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        **kwargs,
    )


def get_qdrant_client(config: Optional[VectorDBConfig] = None):
    from qdrant_client import QdrantClient

    cfg = config or load_vectordb_config()
    if cfg.provider and cfg.provider != "qdrant":
        raise ValueError(f"VECTORDB_PROVIDER is set to '{cfg.provider}', expected 'qdrant'.")
    url = _require(cfg.url, "VECTORDB_URL or QDRANT_URL")
    return QdrantClient(url=url, api_key=cfg.api_key)
