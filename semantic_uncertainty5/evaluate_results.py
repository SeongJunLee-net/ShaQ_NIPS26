"""Evaluate ShaQ ambiguity scores on AmbigQA results."""

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.config import Config


# The paper reports ShaQ with its total attribution and max-span attribution.
UQ_SCORE_REGISTRY = {
    "max_span_phi": {
        "label": "Max Span φ",
        "extract": lambda record: (
            record.get("max_span_aleatoric")
            if "max_span_aleatoric" in record
            else (
                max(
                    (span.get("phi", 0.0) for span in record["spans"].values()),
                    default=0.0,
                )
                if record.get("spans")
                else record.get("total_aleatoric", 0.0)
            )
        ),
    },
    "total_aleatoric": {
        "label": "Total Aleatoric  Σφ_k",
        "extract": lambda record: record.get("total_aleatoric", 0.0),
    },
}


def _extract_score(record: dict, uq_score: str) -> float:
    entry = UQ_SCORE_REGISTRY.get(uq_score)
    if entry is None:
        choices = ", ".join(UQ_SCORE_REGISTRY)
        raise ValueError(f"Unknown --uq_score {uq_score!r}. Choices: {choices}")
    return float(entry["extract"](record) or 0.0)


def load_records(results_path: str) -> list[dict]:
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}")

    with open(results_path, encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def _save_metrics(metrics: dict, output_file: str, verbose: bool = True) -> None:
    try:
        with open(output_file, "w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=4, ensure_ascii=False)
        if verbose:
            print(f"\n✅ Metrics saved to: {output_file}")
    except Exception as exc:
        if verbose:
            print(f"\n❌ Failed to save metrics to {output_file}: {exc}")


def evaluate_records(
    records: list[dict],
    method: str,
    uq_score: Optional[str] = "max_span_phi",
    output_file: Optional[str] = None,
    verbose: bool = True,
):
    """Evaluate one of the two ShaQ scores over in-memory result records.

    ``method`` remains in the public signature for compatibility with
    ``run.py`` and older callers. ShaQ is now the only supported method.
    """

    del method

    if not records:
        if verbose:
            print("No records found in the results file.")
        return None

    if uq_score is None:
        uq_score = "max_span_phi"
    if uq_score not in UQ_SCORE_REGISTRY:
        _extract_score({}, uq_score)
    metric_name = UQ_SCORE_REGISTRY[uq_score]["label"]
    labels = np.asarray(
        [record.get("is_ambiguous_gold", False) for record in records]
    )
    raw_scores = np.asarray(
        [_extract_score(record, uq_score) for record in records],
        dtype=float,
    )
    scores = np.clip(raw_scores, 0.0, None)
    predicted_statuses = [record.get("status", "unknown") for record in records]

    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
    else:
        if verbose:
            print("\n[INFO] All samples have the same ground-truth ambiguity label.")
            print("       AUROC and AUPRC are not defined and will be skipped.\n")
        auroc, auprc = float("nan"), float("nan")

    gold_ambiguous = int(np.sum(labels))
    gold_unambiguous = len(records) - gold_ambiguous
    predicted_ambiguous = sum(status == "ambiguous" for status in predicted_statuses)
    predicted_clear = sum(status == "clear" for status in predicted_statuses)
    zero_uncertainty = int(np.sum(scores == 0.0))
    non_positive_uncertainty = int(np.sum(raw_scores <= 0.0))

    if verbose:
        print("=" * 60)
        print(f"EVALUATION RESULTS  [{metric_name}]")
        print("=" * 60)
        print(f"Total processed: {len(records)}")
        print(
            f"Gold ambiguous: {gold_ambiguous}   "
            f"Gold unambiguous: {gold_unambiguous}"
        )
        print(
            f"Predicted ambiguous: {predicted_ambiguous}   "
            f"Predicted clear: {predicted_clear}"
        )
        print(f"Non-positive score samples: {non_positive_uncertainty}")
        print(f"Zero-score samples:         {zero_uncertainty}")
        print(f"AUROC : {auroc:.4f}")
        print(f"AUPRC : {auprc:.4f}")
        print("-" * 60)

    if zero_uncertainty == len(records) and verbose:
        print(
            "Diagnostic: all uncertainty scores are 0.0. This usually means "
            "the localizer returned clear for every sample or the aleatoric "
            "estimate collapsed to zero."
        )
        print("-" * 60)

    ambiguous_scores = scores[labels == True]
    unambiguous_scores = scores[labels == False]

    if verbose:
        if len(ambiguous_scores) > 0:
            print(f"Avg score — Ambig   : {np.mean(ambiguous_scores):.4f}")
        if len(unambiguous_scores) > 0:
            print(f"Avg score — Unambig : {np.mean(unambiguous_scores):.4f}")
        print("-" * 60)

    metrics = {
        "total_processed": len(records),
        "gold_ambiguous": gold_ambiguous,
        "gold_unambiguous": gold_unambiguous,
        "predicted_ambiguous": predicted_ambiguous,
        "predicted_clear": predicted_clear,
        "non_positive_uncertainty_samples": non_positive_uncertainty,
        "zero_uncertainty_samples": zero_uncertainty,
        "auroc_aleatoric": float(auroc) if not np.isnan(auroc) else None,
        "auprc_aleatoric": float(auprc) if not np.isnan(auprc) else None,
        "avg_ambig_uncertainty": (
            float(np.mean(ambiguous_scores)) if len(ambiguous_scores) > 0 else None
        ),
        "avg_unambig_uncertainty": (
            float(np.mean(unambiguous_scores))
            if len(unambiguous_scores) > 0
            else None
        ),
        "uq_score_used": metric_name,
    }

    if len(np.unique(labels)) >= 2:
        min_score, max_score = np.min(scores), np.max(scores)
        thresholds = (
            np.linspace(min_score, max_score, 200)
            if min_score != max_score
            else [min_score]
        )

        f1_scores = []
        precisions = []
        recalls = []
        for threshold in thresholds:
            predictions = scores > threshold
            f1_scores.append(f1_score(labels, predictions, zero_division=0))
            precisions.append(precision_score(labels, predictions, zero_division=0))
            recalls.append(recall_score(labels, predictions, zero_division=0))

        best_index = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_index]
        ambiguous_accuracy = (
            np.mean(ambiguous_scores > best_threshold)
            if len(ambiguous_scores) > 0
            else 0.0
        )
        unambiguous_accuracy = (
            np.mean(unambiguous_scores <= best_threshold)
            if len(unambiguous_scores) > 0
            else 0.0
        )

        if verbose:
            print(f"Best Threshold (by F1): {best_threshold:.4f}")
            print(f"Best F1 Score:  {f1_scores[best_index]:.4f}")
            print(f"Best Precision: {precisions[best_index]:.4f}")
            print(f"Best Recall:    {recalls[best_index]:.4f}")
            print("-" * 60)
            print(f"Ambiguous Acc   (w/ Best Thres): {ambiguous_accuracy:.4f}")
            print(f"Unambiguous Acc (w/ Best Thres): {unambiguous_accuracy:.4f}")
            print("=" * 60)

        metrics.update(
            {
                "best_threshold_f1": float(best_threshold),
                "best_f1_score": float(f1_scores[best_index]),
                "best_precision": float(precisions[best_index]),
                "best_recall": float(recalls[best_index]),
                "ambig_accuracy": float(ambiguous_accuracy),
                "unambig_accuracy": float(unambiguous_accuracy),
            }
        )
    else:
        metrics.update(
            {
                "best_threshold_f1": None,
                "best_f1_score": None,
                "best_precision": None,
                "best_recall": None,
                "ambig_accuracy": None,
                "unambig_accuracy": None,
            }
        )

    if output_file is not None:
        _save_metrics(metrics, output_file, verbose=verbose)

    return metrics


def evaluate(
    results_path: str,
    method: str,
    uq_score: Optional[str] = "max_span_phi",
    metrics_filename: str = "metrics.json",
):
    """Load a results JSONL file, evaluate it, and save its metrics."""

    try:
        records = load_records(results_path)
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at {results_path}")
        return None

    output_file = os.path.join(os.path.dirname(results_path), metrics_filename)
    return evaluate_records(
        records,
        method=method,
        uq_score=uq_score,
        output_file=output_file,
        verbose=True,
    )


def compare_scores(results_path: str) -> None:
    """Print both ShaQ score metrics side by side without writing files."""

    try:
        records = load_records(results_path)
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at {results_path}")
        return

    labels = np.asarray(
        [bool(record.get("is_ambiguous_gold", False)) for record in records],
        dtype=bool,
    )
    if len(np.unique(labels)) < 2:
        print("All labels are the same — comparison not meaningful.")
        return

    print("=" * 72)
    print(f"{'UQ Score':<28}  {'AUROC':>6}  {'AUPRC':>6}  {'Best F1':>7}  {'Thres':>7}")
    print("-" * 72)

    for score_key, score_spec in UQ_SCORE_REGISTRY.items():
        raw_scores = np.asarray(
            [_extract_score(record, score_key) for record in records],
            dtype=float,
        )
        scores = np.clip(raw_scores, 0.0, None)
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        min_score, max_score = scores.min(), scores.max()
        thresholds = (
            np.linspace(min_score, max_score, 200)
            if min_score < max_score
            else [min_score]
        )
        f1_scores = [
            f1_score(labels, scores > threshold, zero_division=0)
            for threshold in thresholds
        ]
        best_index = int(np.argmax(f1_scores))
        print(
            f"  {score_spec['label']:<26}  {auroc:>6.4f}  {auprc:>6.4f}  "
            f"{f1_scores[best_index]:>7.4f}  {thresholds[best_index]:>7.4f}"
        )

    print("=" * 72)


def _resolve_results_path(config_path: str, dataset_name: str) -> str | None:
    with open(config_path, encoding="utf-8") as file_obj:
        config = Config(**yaml.safe_load(file_obj))

    model_alias = config.model_name_or_path.split("/")[-1]
    base_dir = os.path.join("results", dataset_name)
    experiment_pattern = (
        f"*_{model_alias}_shapley_m{config.m}_n{config.n}_"
        f"ansT{config.answer_temperature}_repT{config.replace_temperature}_"
        f"maxAns{config.max_new_tokens_answer}_maxRep{config.max_new_tokens_replace}_"
        f"maxClu{config.max_new_tokens_cluster}_maxLoc{config.max_new_tokens_localize}"
    )
    search_pattern = os.path.join(base_dir, experiment_pattern, "results.jsonl")
    matches = sorted(glob.glob(search_pattern), reverse=True)
    if not matches:
        print(f"❌ Error: Could not find any results matching pattern:\n   {search_pattern}")
        return None

    print(f"🔍 Auto-resolved results path to latest: {matches[0]}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AmbigQA results produced by the ShaQ pipeline."
    )
    parser.add_argument(
        "--results_path",
        help="Explicit path to results.jsonl generated by run.py",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml used to find the latest matching result",
    )
    parser.add_argument(
        "--dataset_name",
        default="ambigqa",
        help="Dataset result directory name (default: ambigqa)",
    )
    parser.add_argument(
        "--method",
        default="shapley",
        choices=["shapley"],
        help="UQ method (ShaQ/Shapley only)",
    )
    parser.add_argument(
        "--uq_score",
        default="max_span_phi",
        choices=list(UQ_SCORE_REGISTRY),
        help="ShaQ ambiguity score to evaluate (default: max_span_phi)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare max-span and total ShaQ scores without writing metrics",
    )
    parser.add_argument(
        "--metrics_filename",
        default="metrics.json",
        help="Filename for saved metrics inside the result directory",
    )
    args = parser.parse_args()

    if args.results_path:
        target_path = args.results_path
    elif args.config:
        target_path = _resolve_results_path(args.config, args.dataset_name)
        if target_path is None:
            return
    else:
        parser.error("one of --results_path or --config is required")

    if args.compare:
        compare_scores(target_path)
    else:
        evaluate(
            target_path,
            args.method,
            uq_score=args.uq_score,
            metrics_filename=args.metrics_filename,
        )


if __name__ == "__main__":
    main()
