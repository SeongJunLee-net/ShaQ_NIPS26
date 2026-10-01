"""Utility helpers for the active XML-localized Shapley pipeline."""

import json
import logging
import re
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)
def load_prompt(path: str) -> str:
    """Load a prompt template from *path* and return its content.

    Args:
        path: File path to the prompt template (UTF-8 encoded).

    Returns:
        The full text content of the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json(text: str) -> dict:
    """Extract and parse a JSON object from an LLM response string.

    Tries the following in order:

    1. If the text contains a ``\`\`\`json ... \`\`\``` code block, extract only
       the content between the fences.
    2. Otherwise attempt to parse the entire string as JSON.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed Python dictionary.

    Raises:
        json.JSONDecodeError: If no valid JSON can be found.
    """
    # 1. Try to extract from ```json ... ``` code fence
    """
    ```json
    {
    "answer": "python",
    "score": 0.98
    }
    ```
    Find this fenced JSON block.
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return json.loads(match.group(1), strict=False)

    # 2. Try to find the first { ... } block in the text
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return json.loads(brace_match.group(0), strict=False)

    # 3. Fallback: parse entire string
    return json.loads(text, strict=False)


def retry(
    func: Callable,
    *args: Any,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Call ``func(*args, **kwargs)`` up to *max_retries* times.

    Uses exponential back-off between attempts (1 s, 2 s, 4 s, …).

    Args:
        func: The callable to invoke.
        *args: Positional arguments forwarded to *func*.
        max_retries: Maximum number of attempts.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        The return value of *func* on the first successful call.

    Raises:
        Exception: Re-raises the last exception if all attempts fail.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt  # 1, 2, 4 seconds
            logger.exception(
                "retry: attempt %d/%d failed with exception: %s. Waiting %ds …",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise last_exc
