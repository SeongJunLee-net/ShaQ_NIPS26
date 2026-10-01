"""Singleton OpenRouter client used by the ShaQ inference pipeline."""

import concurrent.futures
import logging
import threading
from typing import List, Optional

from utils.config import Config
from utils.util import retry

logger = logging.getLogger(__name__)


class ModelClient:
    """OpenRouter-backed text generation shared by all ShaQ modules."""

    _instance: Optional["ModelClient"] = None

    def __init__(self, config: Config) -> None:
        self.use_api = True
        self.model_name = config.model_name_or_path
        self.api_key = config.api_key
        self.api_seed = config.api_seed
        self.api_reasoning_effort = config.api_reasoning_effort
        self._usage_lock = threading.Lock()
        self._usage_counters = {
            "generate_invocations": 0,
            "requested_sequences": 0,
            "api_chat_completion_requests": 0,
        }

        if not self.api_key:
            raise ValueError("API key must be provided for OpenRouter inference")

        import openai

        logger.info("Initializing OpenRouter API client for model %s", self.model_name)
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        logger.info("OpenRouter API client initialized successfully")

    @classmethod
    def get_instance(cls, config: Config) -> "ModelClient":
        """Return the process-wide client, creating it on first use."""

        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def tokenize_question(self, question: str) -> List[str]:
        """Retained as an explicit guard for obsolete local-model call sites."""

        del question
        raise NotImplementedError("Tokenization is unavailable in OpenRouter-only mode")

    def reset_usage_counters(self) -> None:
        with self._usage_lock:
            for key in self._usage_counters:
                self._usage_counters[key] = 0

    def get_usage_counters(self) -> dict:
        with self._usage_lock:
            return dict(self._usage_counters)

    def _bump_usage(self, key: str, amount: int = 1) -> None:
        with self._usage_lock:
            self._usage_counters[key] = self._usage_counters.get(key, 0) + amount

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> List[str]:
        """Generate ``n`` responses through OpenRouter.

        OpenRouter providers do not consistently honor multi-choice ``n``.
        Requests therefore use parallel, independently seeded ``n=1`` calls.
        """

        self._bump_usage("generate_invocations")
        self._bump_usage("requested_sequences", n)
        return self._generate_api(prompt, temperature, max_new_tokens, n)

    def _generate_api(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        logger.debug(
            "generate_api | n=%d | temperature=%s | prompt=\n%s",
            n,
            temperature,
            prompt,
        )

        model_short_name = self.model_name.split("/")[-1]
        explicit_reasoning_effort = (
            str(self.api_reasoning_effort).strip().lower()
            if self.api_reasoning_effort is not None
            else None
        )
        if explicit_reasoning_effort in {"", "auto"}:
            explicit_reasoning_effort = None

        reasoning_effort = explicit_reasoning_effort
        if reasoning_effort is None:
            if model_short_name == "gemini-2.5-flash":
                reasoning_effort = "none"
            elif (
                model_short_name.startswith("gpt-5")
                or model_short_name.startswith("gemini-3")
                or model_short_name in {"gpt-5-mini", "gemini-3-flash-preview"}
            ):
                reasoning_effort = "minimal"

        def _call_api_once(sample_index: int = 0) -> str:
            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature if temperature > 0 else 0.0,
                "max_tokens": max_new_tokens,
                "n": 1,
            }
            if reasoning_effort and reasoning_effort != "default":
                request_kwargs["extra_body"] = {
                    "reasoning": {"effort": reasoning_effort}
                }
            if self.api_seed is not None:
                request_kwargs["seed"] = int(self.api_seed) + sample_index

            self._bump_usage("api_chat_completion_requests")
            response = self.client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""

        if n == 1:
            results = [retry(_call_api_once, max_retries=3)]
        else:
            logger.debug(
                "generate_api | using parallel repeated n=1 calls for requested n=%d",
                n,
            )
            max_workers = min(n, 10)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        retry,
                        _call_api_once,
                        sample_index,
                        max_retries=3,
                    )
                    for sample_index in range(n)
                ]
                results = [future.result() for future in futures]

        for decoded in results:
            logger.debug("generate_api | response: %s", decoded)
        return results
