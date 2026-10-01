"""XML-based localizer for ambiguity detection."""

import difflib
import json
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from model.model_client import ModelClient
from utils.config import Config
from utils.format import DetectedSpan, XMLLocalizerResult
from utils.util import extract_json, load_prompt, retry

logger = logging.getLogger(__name__)

AMBIG_TAG_RE = re.compile(r'<ambig id="(\d+)">(.*?)</ambig>', re.DOTALL)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([?.!,;:])", r"\1", text)


def _canonicalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"[‐‑‒–—−]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_ambig_tags(text: str) -> str:
    return AMBIG_TAG_RE.sub(lambda match: match.group(2), text)


def _build_text_diff(original: str, recovered: str, max_lines: int = 12) -> str:
    diff_lines = list(
        difflib.unified_diff(
            [original],
            [recovered],
            fromfile="original",
            tofile="recovered",
            lineterm="",
        )
    )
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... (diff truncated)"]
    return "\n".join(diff_lines)


def _renumber_ambig_tags(tagged_sentence: str) -> Tuple[str, List[str]]:
    old_ids: List[str] = []

    def _replace(match: re.Match[str]) -> str:
        """
        match.group(0) returns the full matched tag.
        match.group(1) returns the first capture group: the numeric id.
        match.group(2) returns the second capture group: the tag contents.
        """
        old_ids.append(match.group(1))
        new_id = len(old_ids)
        return f'<ambig id="{new_id}">{match.group(2)}</ambig>'

    return AMBIG_TAG_RE.sub(_replace, tagged_sentence), old_ids


def parse_localizer_response(response: str, sentence: str) -> XMLLocalizerResult:
    """Parse XML-tag-based localizer output into deterministic span objects."""

    data = extract_json(response)
    status = str(data.get("status", "clear")).strip().lower()
    tagged_sentence_raw = str(data.get("tagged_sentence", sentence))
    tagged_sentence, old_ids_in_order = _renumber_ambig_tags(tagged_sentence_raw)
    recovered_sentence = _strip_ambig_tags(tagged_sentence)

    if _canonicalize_for_match(sentence) != _canonicalize_for_match(recovered_sentence):
        raise ValueError(
            "Localizer changed the original sentence outside XML tags.\n"
            f"Original: {sentence}\n"
            f"Tagged: {tagged_sentence_raw}\n"
            f"Recovered: {_normalize_whitespace(recovered_sentence)}\n"
            "Diff:\n"
            f"{_build_text_diff(_normalize_whitespace(sentence), _normalize_whitespace(recovered_sentence))}"
        )
    """
    Find <ambig id="...">...</ambig> tags in tagged_sentence and return them as re.Match objects.
    """
    matches = list(AMBIG_TAG_RE.finditer(tagged_sentence))
    if status != "ambiguous" or not matches:
        return XMLLocalizerResult(
            tagged_sentence=sentence,
            clean_sentence=sentence,
            spans={},
            status="clear",
        )

    span_meta = data.get("spans", [])
    reason_by_id: Dict[str, str] = {}
    for item in span_meta:
        span_id = str(item.get("id", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if span_id and span_id not in reason_by_id:
            reason_by_id[span_id] = reason

    spans: Dict[str, DetectedSpan] = {}
    for idx, match in enumerate(matches, start=1):
        span_id = str(idx)
        old_id = old_ids_in_order[idx - 1] if idx - 1 < len(old_ids_in_order) else span_id
        span_text = match.group(2)
        reason = reason_by_id.get(
            old_id,
            "Depending on how this span is interpreted, different canonical answers are possible.",
        )
        spans[span_id] = DetectedSpan(
            id=span_id,
            text=span_text,
            reason=reason,
        )

    return XMLLocalizerResult(
        tagged_sentence=tagged_sentence,
        clean_sentence=sentence,
        spans=spans,
        status="ambiguous",
    )


def localize(
    x: str,
    tokens: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> XMLLocalizerResult:
    """Run the XML localizer.

    ``tokens`` is accepted only for backward compatibility with older call sites.
    """

    if config is None:
        if isinstance(tokens, Config):
            config = tokens
            tokens = None
        else:
            raise TypeError("config must be provided to localize().")

    del tokens

    prompt_template = load_prompt(config.localizer_prompt_path)
    filled_prompt = prompt_template.replace("{x}", x)
    client = ModelClient.get_instance(config)

    def _call_and_parse() -> XMLLocalizerResult:
        responses = client.generate(
            prompt=filled_prompt,
            temperature=0.0,
            max_new_tokens=config.max_new_tokens_localize,
            n=1,
        )
        return parse_localizer_response(responses[0], x)

    try:
        result = retry(_call_and_parse, max_retries=3)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Localizer fallback to clear after malformed output for query=%r: %s",
            x,
            exc,
        )
        result = XMLLocalizerResult(
            tagged_sentence=x,
            clean_sentence=x,
            spans={},
            status="clear",
        )
    logger.info("Localizer: status=%s spans=%d", result.status, len(result.spans))
    return result


def localize_xml(
    x: str,
    config: Config,
) -> XMLLocalizerResult:
    """Explicit alias for XML-based localization."""

    return localize(x=x, config=config)
