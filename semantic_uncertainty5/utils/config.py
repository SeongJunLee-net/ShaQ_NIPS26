"""Configuration for the OpenRouter-backed AmbigQA ShaQ pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """ShaQ runtime settings loaded from ``config.yaml``."""

    method: str = "shapley"

    # OpenRouter
    model_name_or_path: str = "openai/gpt-4"
    use_api: bool = True
    api_key: Optional[str] = None
    api_seed: Optional[int] = 42
    api_reasoning_effort: Optional[str] = None
    max_workers: int = 5

    # Paper configuration
    m: int = 3
    n: int = 5
    answer_temperature: float = 0.7
    replace_temperature: float = 0.9

    # Generation limits
    max_new_tokens_answer: int = 256
    max_new_tokens_replace: int = 512
    max_new_tokens_cluster: int = 2048
    max_new_tokens_localize: int = 512

    # ShaQ prompts
    localizer_prompt_path: str = "prompts/localizer.txt"
    premise_generator_prompt_path: str = "prompts/premise_generator.txt"
    answerer_prompt_path: str = "prompts/answerer.txt"
    clusterer_prompt_path: str = "prompts/clusterer.txt"

    # Exact Shapley enumeration is exponential in the localized span count.
    max_spans_for_shapley: int = 5
