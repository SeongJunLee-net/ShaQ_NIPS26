"""Premise-conditioned answer sampling for ShaQ."""

import copy
import logging
import time
from typing import List

from utils.config import Config
from model.model_client import ModelClient
from utils.util import load_prompt

logger = logging.getLogger(__name__)


LAST_SAMPLE_DEBUG_TRACE: dict = {}


def _set_last_sample_trace(trace: dict) -> None:
    global LAST_SAMPLE_DEBUG_TRACE
    LAST_SAMPLE_DEBUG_TRACE = trace


def get_last_sample_trace() -> dict:
    return copy.deepcopy(LAST_SAMPLE_DEBUG_TRACE)


def sample_conditioned(
    question: str,
    premise_blocks: List[str],
    n: int,
    config: Config,
) -> List[List[str]]:
    """Sample answers conditioned on the original question plus premise blocks."""

    t0 = time.perf_counter()
    logger.info(
        "AnswerSampler: 1 question × %d premise block(s) × %d answers each",
        len(premise_blocks),
        n,
    )

    prompt_template = load_prompt(config.answerer_prompt_path)
    client = ModelClient.get_instance(config)
    all_answers: List[List[str]] = []
    sample_debug = {
        "mode": "sample_conditioned",
        "question": question,
        "requested_n": n,
        "items": [],
    }

    for idx, premises in enumerate(premise_blocks):
        premise_text = premises.strip() if premises.strip() else "None."
        filled_prompt = (
            prompt_template
            .replace("{clean_sentence}", question)
            .replace("{premises}", premise_text)
        )
        logger.debug("AnswerSampler: premise_block[%d]=%r", idx, premise_text)

        answers = client.generate(
            prompt=filled_prompt,
            temperature=config.answer_temperature,
            max_new_tokens=config.max_new_tokens_answer,
            n=n,
        )
        all_answers.append(answers)
        sample_debug["items"].append(
            {
                "index": idx,
                "premises": premise_text,
                "prompt": filled_prompt,
                "requested_n": n,
                "returned_n": len(answers),
                "answers": answers,
            }
        )

    elapsed = time.perf_counter() - t0
    logger.info("AnswerSampler: conditioned sampling done in %.2fs", elapsed)
    sample_debug["elapsed_sec"] = elapsed
    _set_last_sample_trace(sample_debug)
    return all_answers
