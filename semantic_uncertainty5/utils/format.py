"""Data models used by the current XML-localized Shapley pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DetectedSpan:
    """A span localized via XML tags inside the original sentence.

    Attributes:
        id: XML tag identifier such as ``"1"``.
        text: Raw text wrapped by the XML tag.
        reason: Why this span is ambiguous.
    """

    id: str
    text: str
    reason: str


@dataclass
class XMLLocalizerResult:
    """Output of the XML-based localizer used by the conditional QA pipeline.

    Attributes:
        tagged_sentence: Original sentence with ``<ambig id="k">`` tags inserted.
        clean_sentence: Original sentence with XML tags removed.
        spans: Mapping from XML id to :class:`DetectedSpan`.
        status: ``"ambiguous"`` when spans are present, otherwise ``"clear"``.
    """

    tagged_sentence: str
    clean_sentence: str
    spans: Dict[str, DetectedSpan] = field(default_factory=dict)
    status: str = "clear"


@dataclass
class PremiseGeneration:
    """Cached premise candidates for one detected span."""

    span_id: str
    span_text: str
    premises: List[str] = field(default_factory=list)


@dataclass
class ShapleySpanResult:
    """Shapley-based aleatoric uncertainty result for a single span.

    Attributes:
        span_text: Original span text.
        reason: Why the span is ambiguous (from Localizer).
        phi: Shapley value — fair share of total aleatoric uncertainty.
    """

    token_indices: List[int | str]
    span_text: str
    reason: str
    phi: float
    err: float = 0.0
    premises: List[str] = field(default_factory=list)
    replacements: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ShapleyUQResult:
    """Full output of the Shapley-based UQ pipeline for one input query.

    Key property (efficiency):
        total_aleatoric ≈ sum of all span phi values
        total_uncertainty ≈ total_aleatoric + epistemic

    Attributes:
        status: ``"clear"`` if the query has no ambiguity, else ``"ambiguous"``.
        sentence: Original input query *x*.
        spans: Mapping from span id to :class:`ShapleySpanResult`.
        H_cache: Mapping from subset key string to H(Y|X, C_S).
                 Keys are stringified frozensets, e.g. ``"{}"`` or ``"{0, 1}"``.
        total_uncertainty: H(Y|X) — entropy of original query.
        total_aleatoric: I(Y; C|X) = H(Y|X) - H(Y|X, C_N).
        epistemic: H(Y|X, C_N) — residual entropy after full clarification.
        max_span_aleatoric: max_{k} φ_k — the single span with the highest
                            Shapley value. Used as a conservative UQ signal
                            that is robust to noisy/low-contribution spans.
                            0.0 for ``"clear"`` queries (no spans).
    """

    status: str
    sentence: str
    spans: Dict[int, ShapleySpanResult] = field(default_factory=dict)
    H_cache: Dict[str, float] = field(default_factory=dict)
    total_uncertainty: float = 0.0
    total_aleatoric: float = 0.0
    epistemic: float = 0.0
    global_err: float = 0.0
    max_span_aleatoric: float = 0.0
    debug_traces: Dict[str, dict] = field(default_factory=dict)
