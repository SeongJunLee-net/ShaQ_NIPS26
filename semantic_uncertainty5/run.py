"""Run the AmbigQA ShaQ experiment reported in Table 1 of the paper."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import random
import re
import tarfile
import time
from itertools import product
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import yaml

from model.answer_sampler import get_last_sample_trace, sample_conditioned
from model.localizer import localize_xml
from model.model_client import ModelClient
from model.replacer import generate_premise_cache
from utils.config import Config
from utils.format import DetectedSpan, PremiseGeneration, ShapleySpanResult, ShapleyUQResult, XMLLocalizerResult
from utils.semantic_entropy import (
    cluster_answers_once_with_trace,
    compute_entropy_from_cluster_ids_with_unknown_redistribution,
    compute_entropy_from_preclustered_answers_with_trace,
    compute_se,
    compute_se_with_trace,
)
from utils.shapley import all_subsets, compute_shapley_values


BASE_DIR = Path(__file__).resolve().parent
PROMPT_FIELDS = (
    "localizer_prompt_path",
    "premise_generator_prompt_path",
    "answerer_prompt_path",
    "clusterer_prompt_path",
)
SNAPSHOT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "dataset_cache",
    "results",
    "snapshots",
}
SNAPSHOT_EXCLUDED_SUFFIXES = (".pyc", ".pyo")
SNAPSHOT_EXCLUDED_FILE_PREFIXES = (".env",)


def resolve_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve paths from either the working directory or project directory."""
    if not path_str:
        return path_str
    path = Path(path_str).expanduser()
    if path.is_absolute() or path.exists():
        return str(path.resolve())
    return str((BASE_DIR / path).resolve())


def load_config(config_path: str) -> Config:
    with open(config_path, encoding="utf-8") as handle:
        config = Config(**(yaml.safe_load(handle) or {}))
    for field_name in PROMPT_FIELDS:
        setattr(config, field_name, resolve_path(getattr(config, field_name)))
    return config


def config_for_save(config: Config) -> dict:
    """Serialize configuration without persisting the API credential."""
    payload = dataclasses.asdict(config)
    payload["api_key"] = None
    forced_spans = getattr(config, "forced_spans", None)
    if forced_spans:
        payload["forced_spans"] = forced_spans
    return payload


def should_exclude_from_snapshot(path: Path) -> bool:
    relative_parts = path.relative_to(BASE_DIR).parts
    return (
        any(part in SNAPSHOT_EXCLUDED_DIRS for part in relative_parts)
        or path.name.endswith(SNAPSHOT_EXCLUDED_SUFFIXES)
        or path.name.startswith(SNAPSHOT_EXCLUDED_FILE_PREFIXES)
    )


def iter_snapshot_paths(root: Path):
    for child in sorted(root.iterdir()):
        if should_exclude_from_snapshot(child):
            continue
        yield child
        if child.is_dir():
            yield from iter_snapshot_paths(child)


def archive_code_snapshot(experiment_dir: Path) -> Path:
    snapshot_path = experiment_dir / "code_snapshot.tar.gz"
    with tarfile.open(snapshot_path, "w:gz") as archive:
        for path in iter_snapshot_paths(BASE_DIR):
            archive.add(
                path,
                arcname=Path(BASE_DIR.name) / path.relative_to(BASE_DIR),
                recursive=False,
            )
    print(f"[run.py] Code snapshot: {snapshot_path}")
    return snapshot_path


def _sort_span_id(span_id: str):
    return (0, int(span_id)) if span_id.isdigit() else (1, span_id)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_forced_span(sentence: str, span_text: str) -> Optional[tuple[int, int]]:
    if not span_text:
        return None
    match = re.search(rf"(?<!\w){re.escape(span_text)}(?!\w)", sentence, re.IGNORECASE)
    if match:
        return match.span()
    start = sentence.lower().find(span_text.lower())
    return None if start < 0 else (start, start + len(span_text))


def _tag_sentence_with_forced_spans(sentence: str, spans: dict[str, DetectedSpan]) -> str:
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for span_id, span in sorted(spans.items(), key=lambda item: _sort_span_id(item[0])):
        match = _find_forced_span(sentence, span.text)
        if match is None:
            continue
        start, end = match
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        matches.append((start, end, span_id))

    pieces: list[str] = []
    cursor = 0
    for start, end, span_id in sorted(matches):
        pieces.extend((sentence[cursor:start], f'<ambig id="{span_id}">', sentence[start:end], "</ambig>"))
        cursor = end
    pieces.append(sentence[cursor:])
    return "".join(pieces)


def forced_localize(sentence: str, config: Config) -> Optional[XMLLocalizerResult]:
    forced_spans = getattr(config, "forced_spans", None) or []
    if not forced_spans:
        return None

    spans: dict[str, DetectedSpan] = {}
    seen: set[str] = set()
    for raw_span in forced_spans:
        if isinstance(raw_span, str):
            text = raw_span.strip()
            reason = "User-forced ambiguity span."
        else:
            text = str(raw_span.get("text", "")).strip()
            reason = str(raw_span.get("reason", "")).strip() or "User-forced ambiguity span."
        if not text:
            raise ValueError("Forced spans must include non-empty text.")
        if _find_forced_span(sentence, text) is None:
            raise ValueError(f"Forced span {text!r} was not found in query {sentence!r}.")
        normalized = _normalize_for_match(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        span_id = str(len(spans) + 1)
        spans[span_id] = DetectedSpan(id=span_id, text=text, reason=reason)

    return XMLLocalizerResult(
        tagged_sentence=_tag_sentence_with_forced_spans(sentence, spans),
        clean_sentence=sentence,
        spans=spans,
        status="ambiguous",
    )


def get_localized(sentence: str, config: Config) -> XMLLocalizerResult:
    return forced_localize(sentence, config) or localize_xml(sentence, config)


def serialize_subset(subset: FrozenSet[str]) -> str:
    if not subset:
        return "{}"
    return "{" + ", ".join(sorted(subset, key=_sort_span_id)) + "}"


def build_condition_context(question: str, premises: str) -> str:
    return f"Question: {question}\nAssumptions:\n{premises.strip() or 'None.'}"


def build_premise_block(
    assignment: Dict[str, int],
    localized_spans: dict[str, DetectedSpan],
    premise_cache: dict[str, PremiseGeneration],
) -> str:
    lines: list[str] = []
    for span_id in sorted(assignment, key=_sort_span_id):
        span = localized_spans[span_id]
        premise = premise_cache[span_id].premises[assignment[span_id]]
        lines.append(f'- Span {span_id} ("{span.text or "[insertion point]"}"): {premise}')
    return "\n".join(lines)


def iter_premise_assignments(span_ids, premise_cache):
    ordered_ids = sorted(span_ids, key=_sort_span_id)
    if not ordered_ids:
        yield {}
        return
    premise_ranges = [range(len(premise_cache[span_id].premises)) for span_id in ordered_ids]
    for premise_indices in product(*premise_ranges):
        yield dict(zip(ordered_ids, premise_indices))


def make_leaf_key(ordered_span_ids, assignment: Dict[str, int]):
    return tuple((span_id, assignment[span_id]) for span_id in ordered_span_ids)


def serialize_leaf_key(leaf_key) -> str:
    return "(" + ", ".join(f"{span_id}:{premise_idx}" for span_id, premise_idx in leaf_key) + ")"


def generate_leaf_cache(
    question: str,
    ordered_span_ids,
    localized_spans: dict[str, DetectedSpan],
    premise_cache: dict[str, PremiseGeneration],
    config: Config,
):
    assignments = list(iter_premise_assignments(ordered_span_ids, premise_cache))
    premise_blocks = [build_premise_block(item, localized_spans, premise_cache) for item in assignments]
    answers_matrix = sample_conditioned(
        question=question,
        premise_blocks=premise_blocks,
        n=config.n,
        config=config,
    )
    sampling_trace = get_last_sample_trace()
    leaf_cache = {}
    for assignment, premise_block, answers in zip(assignments, premise_blocks, answers_matrix):
        leaf_cache[make_leaf_key(ordered_span_ids, assignment)] = {
            "assignment": assignment,
            "premise_block": premise_block,
            "answers": answers,
        }
    return leaf_cache, sampling_trace


def attach_global_clusters_to_leaf_cache(question: str, leaf_cache, config: Config):
    global_answers: list[str] = []
    answer_refs = []
    for leaf_key, leaf_payload in leaf_cache.items():
        leaf_payload["answer_records"] = []
        for leaf_answer_index, answer in enumerate(leaf_payload["answers"]):
            global_index = len(global_answers)
            global_answers.append(answer)
            answer_refs.append((leaf_key, leaf_answer_index, global_index))
    if not global_answers:
        raise ValueError("No answers were produced for global semantic clustering.")

    cluster_ids, cluster_trace = cluster_answers_once_with_trace(
        sentence=question,
        answers=global_answers,
        config=config,
        cluster_source="global_leaf_clusterer_once",
    )
    for leaf_key, leaf_answer_index, global_index in answer_refs:
        leaf_cache[leaf_key]["answer_records"].append(
            {
                "leaf_answer_index": leaf_answer_index,
                "global_answer_index": global_index,
                "answer": global_answers[global_index],
                "cluster_id": cluster_ids[global_index],
            }
        )
    return cluster_trace


def measure_entropy_for_subset(
    question: str,
    subset_ids: FrozenSet[str],
    localized_spans: dict[str, DetectedSpan],
    premise_cache: dict[str, PremiseGeneration],
    leaf_cache,
    collect_debug_trace: bool = False,
) -> tuple[float, Optional[dict]]:
    subset_key = serialize_subset(subset_ids)
    assignments = list(iter_premise_assignments(subset_ids, premise_cache))
    premise_blocks = [build_premise_block(item, localized_spans, premise_cache) for item in assignments]
    entropies: list[float] = []
    sample_traces: list[dict] = []

    for sample_index, (assignment, premise_block) in enumerate(zip(assignments, premise_blocks)):
        pooled_answers: list[str] = []
        pooled_cluster_ids: list[int] = []
        pooled_global_indices: list[int] = []
        matched_leaf_keys: list[str] = []
        for leaf_key, leaf_payload in leaf_cache.items():
            leaf_assignment = leaf_payload["assignment"]
            if not all(leaf_assignment[span_id] == premise_idx for span_id, premise_idx in assignment.items()):
                continue
            matched_leaf_keys.append(serialize_leaf_key(leaf_key))
            for answer_record in leaf_payload["answer_records"]:
                pooled_answers.append(answer_record["answer"])
                pooled_cluster_ids.append(answer_record["cluster_id"])
                pooled_global_indices.append(answer_record["global_answer_index"])

        if not pooled_cluster_ids:
            raise ValueError(f"No pooled answers found for subset {subset_key}.")
        if collect_debug_trace:
            entropy, trace = compute_entropy_from_preclustered_answers_with_trace(
                sentence=build_condition_context(question, premise_block),
                answers=pooled_answers,
                cluster_ids=pooled_cluster_ids,
                cluster_source="global_cluster_pooling",
                clusterer_output=None,
            )
            sample_traces.append(
                {
                    "sample_index": sample_index,
                    "premises": premise_block,
                    "matched_leaf_keys": matched_leaf_keys,
                    "global_answer_indices": pooled_global_indices,
                    "global_cluster_ids": pooled_cluster_ids,
                    "pooled_answer_count": len(pooled_answers),
                    **trace,
                }
            )
        else:
            entropy, _ = compute_entropy_from_cluster_ids_with_unknown_redistribution(
                pooled_answers, pooled_cluster_ids
            )
        entropies.append(entropy)

    expected_entropy = sum(entropies) / len(entropies)
    trace = None
    if collect_debug_trace:
        trace = {"subset": subset_key, "expected_entropy": expected_entropy, "sample_traces": sample_traces}
    return expected_entropy, trace


def compute_shapley(question: str, config: Config, dump_answer_traces: bool = False) -> ShapleyUQResult:
    localized = get_localized(question, config)

    if localized.status == "clear":
        answers = sample_conditioned(
            question=localized.clean_sentence, premise_blocks=[""], n=config.n, config=config
        )[0]
        context = build_condition_context(localized.clean_sentence, "")
        if dump_answer_traces:
            entropy, trace = compute_se_with_trace(
                sentence=context,
                answers=answers,
                config=config,
            )
            debug_traces = {
                "{}": {
                    "subset": "{}",
                    "expected_entropy": entropy,
                    "sample_traces": [
                        {
                            "sample_index": 0,
                            "premises": "",
                            "matched_leaf_keys": ["()"],
                            "pooled_answer_count": len(answers),
                            **trace,
                        }
                    ],
                }
            }
        else:
            entropy = compute_se(sentence=context, answers=answers, config=config)
            debug_traces = {}
        return ShapleyUQResult(
            status="clear",
            sentence=localized.clean_sentence,
            H_cache={"{}": entropy},
            total_uncertainty=entropy,
            total_aleatoric=0.0,
            epistemic=entropy,
            max_span_aleatoric=0.0,
            global_err=0.0,
            debug_traces=debug_traces,
        )

    ordered_span_ids = sorted(localized.spans, key=_sort_span_id)[: config.max_spans_for_shapley]
    spans = {span_id: localized.spans[span_id] for span_id in ordered_span_ids}
    premise_cache = generate_premise_cache(
        clean_sentence=localized.clean_sentence, spans=spans, m=config.m, config=config
    )
    leaf_cache, sampling_trace = generate_leaf_cache(
        question=localized.clean_sentence,
        ordered_span_ids=ordered_span_ids,
        localized_spans=spans,
        premise_cache=premise_cache,
        config=config,
    )
    cluster_trace = attach_global_clusters_to_leaf_cache(localized.clean_sentence, leaf_cache, config)

    players = set(spans)
    entropy_cache: Dict[FrozenSet[str], float] = {}
    debug_traces: Dict[str, dict] = {}
    if dump_answer_traces:
        if getattr(config, "forced_spans", None):
            debug_traces["__forced_localizer__"] = {
                "tagged_sentence": localized.tagged_sentence,
                "spans": dataclasses.asdict(localized)["spans"],
            }
        debug_traces["__answer_sampling__"] = sampling_trace
        debug_traces["__global_cluster__"] = cluster_trace

    for subset in all_subsets(players):
        entropy, trace = measure_entropy_for_subset(
            question=localized.clean_sentence,
            subset_ids=subset,
            localized_spans=spans,
            premise_cache=premise_cache,
            leaf_cache=leaf_cache,
            collect_debug_trace=dump_answer_traces,
        )
        entropy_cache[subset] = entropy
        if trace is not None:
            debug_traces[serialize_subset(subset)] = trace

    shapley_values = compute_shapley_values(entropy_cache, players)
    total_uncertainty = entropy_cache[frozenset()]
    epistemic = entropy_cache[frozenset(players)]
    raw_aleatoric = total_uncertainty - epistemic
    if raw_aleatoric < 0:
        print("\n" + "=" * 50)
        print("[WARNING] negative aleatoric value detected")
        print(f"Question: {localized.clean_sentence}")
        print(f"Total Uncertainty: {total_uncertainty}")
        print(f"Epistemic (Expected Entropy): {epistemic}")
        print(f"Raw Aleatoric: {raw_aleatoric}")
        print("=" * 50 + "\n")

    phi_sum = sum(shapley_values.values())
    if abs(abs(phi_sum) - abs(raw_aleatoric)) > 1e-5:
        print("\n" + "*" * 50)
        print("[WARNING] Shapley sum mismatch detected!")
        print(f"Question: {localized.clean_sentence}")
        print(f"Sum of Phis: {phi_sum}")
        print(f"Raw Aleatoric: {raw_aleatoric}")
        print(f"Diff: {abs(abs(phi_sum) - abs(raw_aleatoric))}")
        print("*" * 50 + "\n")

    span_results: Dict[int, ShapleySpanResult] = {}
    for span_id in ordered_span_ids:
        singleton_entropy = entropy_cache.get(frozenset({span_id}), total_uncertainty)
        err = (total_uncertainty - singleton_entropy) / total_uncertainty if total_uncertainty > 0 else 0.0
        span = spans[span_id]
        span_results[int(span_id)] = ShapleySpanResult(
            token_indices=[f"ambig:{span_id}"],
            span_text=span.text or "[insertion point]",
            reason=span.reason,
            phi=shapley_values[span_id],
            err=err,
            premises=premise_cache[span_id].premises,
        )

    return ShapleyUQResult(
        status="ambiguous",
        sentence=localized.clean_sentence,
        spans=span_results,
        H_cache={serialize_subset(key): value for key, value in entropy_cache.items()},
        total_uncertainty=total_uncertainty,
        total_aleatoric=max(0.0, raw_aleatoric),
        epistemic=epistemic,
        global_err=raw_aleatoric / total_uncertainty if total_uncertainty > 0 else 0.0,
        max_span_aleatoric=max(shapley_values.values(), default=0.0),
        debug_traces=debug_traces,
    )


def run_query(question: str, config: Config, dump_answer_traces: bool = False) -> dict:
    return dataclasses.asdict(compute_shapley(question, config, dump_answer_traces))


def select_dataset_entries(entries, max_samples: Optional[int], sampling_seed: int):
    if max_samples is None or max_samples >= len(entries):
        return entries
    selected = random.Random(sampling_seed).sample(list(enumerate(entries)), k=max_samples)
    selected.sort(key=lambda item: item[0])
    return [entry for _, entry in selected]


def _default_destination(config: Config) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_alias = config.model_name_or_path.split("/")[-1]
    name = (
        f"{timestamp}_{model_alias}_shapley_m{config.m}_n{config.n}_"
        f"ansT{config.answer_temperature}_repT{config.replace_temperature}_"
        f"maxAns{config.max_new_tokens_answer}_maxRep{config.max_new_tokens_replace}_"
        f"maxClu{config.max_new_tokens_cluster}_maxLoc{config.max_new_tokens_localize}"
    )
    output_dir = Path("results") / "ambigqa" / name
    suffix = 1
    while output_dir.exists():
        output_dir = output_dir.with_name(f"{name}_{suffix}")
        suffix += 1
    return output_dir / "results.jsonl"


def run_dataset(
    dataset_path: str,
    config: Config,
    output_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    sampling_seed: int = 42,
    dump_answer_traces: bool = False,
) -> Path:
    from utils.dataset import AmbigQADataset

    entries = select_dataset_entries(AmbigQADataset(dataset_path).data, max_samples, sampling_seed)
    destination = Path(output_path) if output_path else _default_destination(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = destination.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    trace_path = logs_dir / "trace.jsonl"
    log_path = logs_dir / "run.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    with open(destination.parent / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_for_save(config), handle, sort_keys=False, allow_unicode=True)

    # Preserve the release behavior: workers append records as they complete.
    import threading

    write_lock = threading.Lock()

    with open(destination, "w", encoding="utf-8"):
        pass
    with open(trace_path, "w", encoding="utf-8"):
        pass

    def process_entry(entry):
        try:
            result = run_query(entry.question, config, dump_answer_traces)
        except Exception as exc:
            result = {
                "status": "error",
                "sentence": entry.question,
                "spans": {},
                "H_cache": {},
                "H_orig": 0.0,
                "total_uncertainty": 0.0,
                "total_aleatoric": 0.0,
                "epistemic": 0.0,
                "global_err": 0.0,
                "max_span_aleatoric": 0.0,
                "debug_traces": {},
                "error": str(exc),
            }
        result["ambigqa_id"] = entry.id
        result["gold_answers"] = entry.gold_answers
        result["is_ambiguous_gold"] = entry.is_ambiguous
        debug_traces = result.pop("debug_traces", {})
        with write_lock:
            with open(destination, "a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            if debug_traces:
                with open(trace_path, "a", encoding="utf-8") as trace_file:
                    trace_file.write(
                        json.dumps(
                            {
                                "ambigqa_id": result["ambigqa_id"],
                                "sentence": result["sentence"],
                                "debug_traces": debug_traces,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(process_entry, entry) for entry in entries]
        for _ in concurrent.futures.as_completed(futures):
            pass

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    print(f"[run.py] Results: {destination}")
    print(f"[run.py] Traces : {trace_path}")
    print(f"[run.py] Log    : {log_path}")
    try:
        archive_code_snapshot(destination.parent)
    except Exception as exc:
        print(f"[run.py] WARNING: failed to archive code snapshot: {exc}")

    return destination


def parse_forced_span_spec(spec: str) -> dict:
    text, separator, reason = spec.partition("::")
    if not text.strip():
        raise ValueError(f"Invalid --forced_span value: {spec!r}")
    return {
        "text": text.strip(),
        "reason": reason.strip() if separator else "User-forced ambiguity span.",
    }


def normalize_forced_span_payload(payload) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, str):
        return [parse_forced_span_spec(payload)]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Forced span payload must be a string, object, or list.")
    normalized: list[dict] = []
    for item in payload:
        if isinstance(item, str):
            normalized.append(parse_forced_span_spec(item))
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError(f"Invalid forced span object without text: {item!r}")
            normalized.append(
                {"text": text, "reason": str(item.get("reason", "")).strip() or "User-forced ambiguity span."}
            )
        else:
            raise ValueError(f"Invalid forced span item: {item!r}")
    return normalized


def load_forced_spans_from_args(args: argparse.Namespace) -> list[dict]:
    forced_spans = []
    if args.forced_spans_path:
        with open(args.forced_spans_path, "r", encoding="utf-8") as handle:
            forced_spans.extend(normalize_forced_span_payload(json.load(handle)))
    if args.forced_spans_json:
        forced_spans.extend(normalize_forced_span_payload(json.loads(args.forced_spans_json)))
    for spec in args.forced_span or []:
        forced_spans.append(parse_forced_span_spec(spec))

    deduplicated = []
    seen = set()
    for span in forced_spans:
        key = _normalize_for_match(span["text"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(span)
    return deduplicated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AmbigQA ShaQ runner (paper Table 1)")
    parser.add_argument("--config", default="utils/config.yaml", help="ShaQ YAML config")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query", help="Run one question (low-cost smoke test)")
    inputs.add_argument("--dataset", help="AmbigQA dev JSON or JSONL")
    parser.add_argument("--output", help="Output JSON for a query or JSONL for a dataset")
    parser.add_argument("--max_samples", type=int, help="Random dataset sample size")
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Write both Table 1 metrics")
    parser.add_argument("--dump_answer_traces", action="store_true")
    parser.add_argument("--model", help="OpenRouter model override")
    parser.add_argument("--max_workers", type=int)
    parser.add_argument("--m", type=int, help="Premises per localized span")
    parser.add_argument("--n", "--answer_sample_n", dest="n", type=int, help="Answers per leaf condition")
    parser.add_argument("--answer_temperature", type=float)
    parser.add_argument("--replace_temperature", type=float)
    parser.add_argument("--max_new_tokens_answer", type=int)
    parser.add_argument("--max_new_tokens_replace", type=int)
    parser.add_argument("--max_new_tokens_cluster", type=int)
    parser.add_argument("--max_new_tokens_localize", type=int)
    parser.add_argument("--max_spans_for_shapley", type=int)
    parser.add_argument(
        "--forced_span",
        action="append",
        default=[],
        metavar="TEXT[::REASON]",
        help="Bypass localization with an exact query span; repeat for multiple spans.",
    )
    parser.add_argument("--forced_spans_json", help="JSON list of forced span strings/objects")
    parser.add_argument("--forced_spans_path", help="JSON file containing forced spans")
    return parser.parse_args()


def _apply_overrides(config: Config, args: argparse.Namespace) -> None:
    mapping = {
        "model": "model_name_or_path",
        "max_workers": "max_workers",
        "m": "m",
        "n": "n",
        "answer_temperature": "answer_temperature",
        "replace_temperature": "replace_temperature",
        "max_new_tokens_answer": "max_new_tokens_answer",
        "max_new_tokens_replace": "max_new_tokens_replace",
        "max_new_tokens_cluster": "max_new_tokens_cluster",
        "max_new_tokens_localize": "max_new_tokens_localize",
        "max_spans_for_shapley": "max_spans_for_shapley",
    }
    for argument, field_name in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(config, field_name, value)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    _apply_overrides(config, args)
    forced_spans = load_forced_spans_from_args(args)
    if forced_spans:
        config.forced_spans = forced_spans

    config.api_key = os.environ.get("OPENROUTER_API_KEY")
    if not config.api_key:
        raise ValueError("Set OPENROUTER_API_KEY in the environment.")
    ModelClient.get_instance(config)

    if args.query:
        if args.evaluate:
            raise ValueError("--evaluate requires --dataset.")
        result = run_query(args.query, config, args.dump_answer_traces)
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return

    destination = run_dataset(
        dataset_path=resolve_path(args.dataset),
        config=config,
        output_path=args.output,
        max_samples=args.max_samples,
        sampling_seed=args.sampling_seed,
        dump_answer_traces=args.dump_answer_traces,
    )
    if args.evaluate:
        from evaluate_results import evaluate

        max_metrics = evaluate(
            str(destination), "shapley", uq_score="max_span_phi", metrics_filename="metrics_max_span_phi.json"
        )
        evaluate(
            str(destination), "shapley", uq_score="total_aleatoric", metrics_filename="metrics_total_aleatoric.json"
        )
        summary_path = destination.parent / "metrics.json"
        summary_path.write_text(
            json.dumps(max_metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[run.py] Metrics: {summary_path}")


if __name__ == "__main__":
    main()
