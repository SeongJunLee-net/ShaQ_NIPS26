"""Semantic entropy helpers for globally clustered answer sets."""

import logging
import math
import time
from collections import Counter
from typing import Dict, List, Tuple

from utils.config import Config
from model.model_client import ModelClient
from utils.util import extract_json, load_prompt, retry

logger = logging.getLogger(__name__)

UNKNOWN_WORD_POOL = [
    "unknown",
    "I don't",
    "did not",
    "Not specified",
    "cannot be determined",
    "No Answer",
    "No final answer",
    "I do not",
    "N/A",
    "No information",
    "It depends on",
    "I cannot",
    "I can't ",
    "I am unable",
    "don't know",
    "No answer",
    "Nobody",
    "enough information",
    "specific information",
    "There is no",
    "No specific",
    "not provided",
    "None",
    "No character",
    "No output",
    "Cannot answer",
    "Unavailable",
    "TBD",
    "To Be Determined",
    "I am asking about",
    "current year",
    "No translation",
    "depends on",
    "has not",
    "unclear",
    "confusion",
    "incorrect",
    "not aware of",
    "invalid",
    "no one",
]


def is_unknown_answer(answer: str) -> bool:
    lowered = answer.strip().lower()
    unknown_words = [word.lower() for word in UNKNOWN_WORD_POOL]
    return lowered in unknown_words or any(word in lowered for word in unknown_words)


def compute_entropy_from_cluster_ids(cluster_ids: List[int]) -> float:
    """Compute Shannon entropy from a list of pre-assigned cluster ids."""

    if not cluster_ids:
        raise ValueError("cluster_ids must not be empty.")

    n_total = len(cluster_ids)
    counter = Counter(cluster_ids)
    return -sum(
        (count / n_total) * math.log(count / n_total)
        for count in counter.values()
        if count > 0
    )


def compute_entropy_from_cluster_ids_with_unknown_redistribution(
    answers: List[str],
    cluster_ids: List[int],
) -> Tuple[float, Dict[str, object]]:
    if not cluster_ids:
        raise ValueError("cluster_ids must not be empty.")
    if len(answers) != len(cluster_ids):
        raise ValueError("answers and cluster_ids must be aligned with the same length.")

    counter = Counter(cluster_ids)
    unknown_indices = [idx for idx, answer in enumerate(answers) if is_unknown_answer(answer)]
    unknown_cluster_ids = sorted({cluster_ids[idx] for idx in unknown_indices})
    unknown_mass = sum(counter[cluster_id] for cluster_id in unknown_cluster_ids)
    adjusted_counts = {
        cluster_id: float(count)
        for cluster_id, count in counter.items()
        if cluster_id not in unknown_cluster_ids
    }

    if unknown_mass > 0:
        if adjusted_counts:
            redistributed = unknown_mass / len(adjusted_counts)
            for cluster_id in adjusted_counts:
                adjusted_counts[cluster_id] += redistributed
        else:
            adjusted_counts = {}

    total = sum(adjusted_counts.values())
    if total <= 0:
        entropy = 0.0
    else:
        entropy = -sum(
            (count / total) * math.log(count / total)
            for count in adjusted_counts.values()
            if count > 0
        )

    trace = {
        "unknown_policy": "redistribute_to_non_unknown_clusters",
        "unknown_answer_indices": unknown_indices,
        "unknown_cluster_ids": [int(cluster_id) for cluster_id in unknown_cluster_ids],
        "unknown_mass": float(unknown_mass),
        "adjusted_cluster_sizes": {
            str(cluster_id): float(count)
            for cluster_id, count in sorted(adjusted_counts.items())
        },
        "raw_cluster_sizes": {
            str(cluster_id): int(count)
            for cluster_id, count in sorted(counter.items())
        },
    }
    return entropy, trace


def compute_entropy_from_preclustered_answers_with_trace(
    sentence: str,
    answers: List[str],
    cluster_ids: List[int],
    *,
    cluster_source: str,
    clusterer_output: dict | None = None,
) -> Tuple[float, Dict[str, object]]:
    """Compute entropy and a debug trace from precomputed cluster ids."""

    if len(answers) != len(cluster_ids):
        raise ValueError(
            "answers and cluster_ids must be aligned with the same length."
        )

    normalized_answers = [a.strip().lower() for a in answers]
    counter = Counter(cluster_ids)
    entropy, unknown_trace = compute_entropy_from_cluster_ids_with_unknown_redistribution(
        answers,
        cluster_ids,
    )
    trace = {
        "sentence": sentence,
        "answers": answers,
        "normalized_answers": normalized_answers,
        "clusters": {
            str(answer_idx): int(cluster_id)
            for answer_idx, cluster_id in enumerate(cluster_ids)
        },
        "cluster_sizes": {
            str(cluster_id): count for cluster_id, count in sorted(counter.items())
        },
        "entropy": entropy,
        "raw_entropy": compute_entropy_from_cluster_ids(cluster_ids),
        "unknown_adjustment": unknown_trace,
        "cluster_source": cluster_source,
        "clusterer_output": clusterer_output,
    }
    return entropy, trace


def cluster_answers_once_with_trace(
    sentence: str,
    answers: List[str],
    config: Config,
    *,
    cluster_source: str = "llm_clusterer",
) -> Tuple[List[int], Dict[str, object]]:
    """Cluster one answer list exactly once and return aligned cluster ids."""

    t0 = time.perf_counter()
    normalized_answers = [a.strip().lower() for a in answers]

    if len(set(normalized_answers)) == 1:
        cluster_ids = [0 for _ in answers]
        entropy, trace = compute_entropy_from_preclustered_answers_with_trace(
            sentence=sentence,
            answers=answers,
            cluster_ids=cluster_ids,
            cluster_source="identical_answers_shortcut",
            clusterer_output=None,
        )
        elapsed = time.perf_counter() - t0
        logger.debug(
            "cluster_answers_once: sentence=%r | all answers identical | H=%.4f nats | %.2fs",
            sentence,
            entropy,
            elapsed,
        )
        return cluster_ids, trace

    prompt_template = load_prompt(config.clusterer_prompt_path)
    answers_numbered = "\n".join(f"{i}. {a}" for i, a in enumerate(answers))
    filled_prompt = (
        prompt_template
        .replace("{question}", sentence)
        .replace("{answers_numbered_list}", answers_numbered)
    )

    client = ModelClient.get_instance(config)

    def _call_and_parse() -> dict:
        responses = client.generate(
            prompt=filled_prompt,
            temperature=0.0,
            max_new_tokens=config.max_new_tokens_cluster,
            n=1,
        )
        return extract_json(responses[0])

    data = retry(_call_and_parse, max_retries=3)
    raw_clusters: dict = data["clusters"]
    cluster_ids: List[int] = []
    for answer_idx in range(len(answers)):
        if str(answer_idx) in raw_clusters:
            cluster_ids.append(int(raw_clusters[str(answer_idx)]))
        elif answer_idx in raw_clusters:
            cluster_ids.append(int(raw_clusters[answer_idx]))
        else:
            raise ValueError(
                f"Clusterer output is missing an assignment for answer index {answer_idx}."
            )

    entropy, trace = compute_entropy_from_preclustered_answers_with_trace(
        sentence=sentence,
        answers=answers,
        cluster_ids=cluster_ids,
        cluster_source=cluster_source,
        clusterer_output=data,
    )

    elapsed = time.perf_counter() - t0
    logger.debug(
        "cluster_answers_once: sentence=%r | clusters=%s | H=%.4f nats | %.2fs",
        sentence,
        trace["cluster_sizes"],
        entropy,
        elapsed,
    )
    return cluster_ids, trace


def compute_se(
    sentence: str,
    answers: List[str],
    config: Config,
) -> float:
    """Compute the Semantic Entropy (SE) of *n* answers for one rewritten sentence.

    The LLM is called with ``temperature=0.0`` (greedy, deterministic) to obtain
    a clustering of the *n* answers.  Shannon entropy is then computed over the
    cluster-size distribution in **nats** (natural logarithm).

    Args:
        sentence: The rewritten sentence that was used to generate *answers*.
                  **Must be the same sentence** the answers were sampled from —
                  not the original query.
        answers: List of *n* answer strings for *sentence*.
        config: Framework configuration.

    Returns:
        Entropy in nats (``float``).
    """
    H, _ = compute_se_with_trace(sentence=sentence, answers=answers, config=config)
    return H


def compute_se_with_trace(
    sentence: str,
    answers: List[str],
    config: Config,
) -> Tuple[float, Dict[str, object]]:
    """Compute semantic entropy and keep the raw answer/cluster trace."""
    _, trace = cluster_answers_once_with_trace(sentence=sentence, answers=answers, config=config)
    return float(trace["entropy"]), trace
