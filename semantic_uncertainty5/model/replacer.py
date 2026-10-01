"""Premise generator for the conditional-QA Shapley pipeline."""

import logging
from typing import Dict, List

from model.model_client import ModelClient
from utils.config import Config
from utils.format import DetectedSpan, PremiseGeneration
from utils.util import extract_json, load_prompt, retry

logger = logging.getLogger(__name__)


def _normalize_premise(text: str) -> str:
    return " ".join(str(text).strip().split())


def _fallback_premises(span: DetectedSpan, m: int) -> List[str]:
    display_text = span.text if span.text else "[insertion point]"
    fallback = f'The ambiguous part "{display_text}" refers to one specific interpretation consistent with the question.'
    return [fallback for _ in range(max(1, m))]


def _parse_premises(response: str, span: DetectedSpan, m: int) -> List[str]:
    data = extract_json(response)
    premises_raw = data.get("premises", [])
    premises: List[str] = []
    seen = set()

    for item in premises_raw:
        premise = _normalize_premise(item)
        if premise and premise not in seen:
            premises.append(premise)
            seen.add(premise)

    if not premises:
        premises = _fallback_premises(span, 1)

    return premises[:m]


def generate_premises_for_span(
    clean_sentence: str,
    span: DetectedSpan,
    m: int,
    config: Config,
) -> PremiseGeneration:
    """Generate up to ``m`` concrete interpretation premises for one span."""

    prompt_template = load_prompt(config.premise_generator_prompt_path)
    display_text = span.text if span.text else "[insertion point]"
    filled_prompt = (
        prompt_template
        .replace("{clean_sentence}", clean_sentence)
        .replace("{span_id}", span.id)
        .replace("{span_text}", display_text)
        .replace("{reason}", span.reason)
        .replace("{m}", str(m))
    )
    client = ModelClient.get_instance(config)

    def _call_and_parse() -> PremiseGeneration:
        responses = client.generate(
            prompt=filled_prompt,
            temperature=config.replace_temperature,
            max_new_tokens=config.max_new_tokens_replace,
            n=1,
        )
        premises = _parse_premises(responses[0], span=span, m=m)
        return PremiseGeneration(
            span_id=span.id,
            span_text=span.text,
            premises=premises,
        )

    try:
        result = retry(_call_and_parse, max_retries=3)
    except Exception:
        logger.exception("Premise generation failed for span %s", span.id)
        result = PremiseGeneration(
            span_id=span.id,
            span_text=span.text,
            premises=_fallback_premises(span, 1),
        )

    logger.info(
        "PremiseGenerator: span=%s produced %d premise(s)",
        span.id,
        len(result.premises),
    )
    return result


def generate_premise_cache(
    clean_sentence: str,
    spans: Dict[str, DetectedSpan],
    m: int,
    config: Config,
) -> Dict[str, PremiseGeneration]:
    """Generate and cache premise candidates for every detected span."""

    cache: Dict[str, PremiseGeneration] = {}
    for span_id in sorted(spans, key=lambda item: int(item) if str(item).isdigit() else item):
        cache[span_id] = generate_premises_for_span(
            clean_sentence=clean_sentence,
            span=spans[span_id],
            m=m,
            config=config,
        )
    return cache
