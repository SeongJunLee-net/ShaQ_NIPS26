"""AmbigQA data loading for the ShaQ Table 1 experiment."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List, Optional


@dataclasses.dataclass
class UQDatasetEntry:
    id: str
    question: str
    gold_answers: List[str]
    is_ambiguous: bool
    ground_truth_spans: List[int] = dataclasses.field(default_factory=list)
    original_tokens: Optional[List[str]] = None


class AmbigQADataset:
    """Load official AmbigQA JSON or the release's preprocessed JSONL."""

    def __init__(self, path: str):
        self.path = path
        self.data = self.load()

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def load(self) -> List[UQDatasetEntry]:
        dataset_path = Path(self.path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        if dataset_path.suffix == ".jsonl":
            with dataset_path.open(encoding="utf-8") as handle:
                raw_data = [json.loads(line) for line in handle if line.strip()]
        else:
            with dataset_path.open(encoding="utf-8") as handle:
                raw_data = json.load(handle)
        if not isinstance(raw_data, list):
            raise ValueError("AmbigQA input must be a JSON list or JSONL records.")

        entries: List[UQDatasetEntry] = []
        for index, record in enumerate(raw_data, start=1):
            entry_id = str(record.get("id", index))
            question = record.get("original_question") or record.get("question", "")
            tokens = record.get("original_tokens")
            ground_truth_spans = record.get("ground_truth_spans", [])
            is_ambiguous = record.get("type") == "multipleQAs"
            gold_answers: list[str] = []
            annotations = record.get("annotations", [])

            if isinstance(annotations, dict):
                if "multipleQAs" in annotations.get("type", []):
                    is_ambiguous = True
                for answers in annotations.get("answer", []):
                    if isinstance(answers, list):
                        gold_answers.extend(answers)
                    elif isinstance(answers, str):
                        gold_answers.append(answers)
                for qa_group in annotations.get("qaPairs", []):
                    if isinstance(qa_group, dict):
                        for answers in qa_group.get("answer", []):
                            if isinstance(answers, list):
                                gold_answers.extend(answers)
                            elif isinstance(answers, str):
                                gold_answers.append(answers)
                    elif isinstance(qa_group, list):
                        for pair in qa_group:
                            if not isinstance(pair, dict) or "answer" not in pair:
                                continue
                            answers = pair["answer"]
                            gold_answers.extend(answers if isinstance(answers, list) else [answers])
            else:
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    annotation_type = annotation.get("type")
                    if annotation_type == "multipleQAs":
                        is_ambiguous = True
                    if "answers" in annotation:
                        gold_answers.extend(annotation["answers"])
                    elif annotation_type == "singleAnswer":
                        gold_answers.extend(annotation.get("answer", []))
                    elif annotation_type == "multipleQAs":
                        for pair in annotation.get("qaPairs", []):
                            if isinstance(pair, dict):
                                gold_answers.extend(pair.get("answer", []))

            unique_gold_answers = list(dict.fromkeys(gold_answers))
            entries.append(
                UQDatasetEntry(
                    id=entry_id,
                    question=question,
                    gold_answers=unique_gold_answers,
                    is_ambiguous=is_ambiguous,
                    ground_truth_spans=ground_truth_spans,
                    original_tokens=tokens,
                )
            )
        return entries


def get_dataset(dataset_path: str) -> AmbigQADataset:
    """Compatibility wrapper retained for older ShaQ callers."""
    return AmbigQADataset(dataset_path)
