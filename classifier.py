"""Reusable NLI-based zero-shot text classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Optional

MODEL_ID = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This text is about {}."
DEFAULT_LABELS = ["sports", "technology", "politics", "education"]


@dataclass
class ClassificationResult:
    sequence: str
    labels: list[str]
    scores: list[float]
    model: str
    revision: Optional[str]
    hypothesis_template: str
    multi_label: bool
    inference_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NewsClassifier:
    """Lazy-loading wrapper around Hugging Face's zero-shot pipeline."""

    def __init__(
        self,
        device: int = -1,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        pipeline_factory: Optional[Callable[..., Any]] = None,
    ):
        self.device = device
        self.model_id = model_id
        self.revision = revision
        self._pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            factory = self._pipeline_factory
            if factory is None:
                from transformers import pipeline
                factory = pipeline

            kwargs = {
                "task": "zero-shot-classification",
                "model": self.model_id,
                "device": self.device,
            }
            if self.revision:
                kwargs["revision"] = self.revision
            self._pipeline = factory(**kwargs)
        return self._pipeline

    @staticmethod
    def _validate(text: str, candidate_labels: list[str]) -> tuple[str, list[str]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Passage cannot be blank.")
        if not isinstance(candidate_labels, (list, tuple)) or len(candidate_labels) < 2:
            raise ValueError("Provide at least two candidate labels.")

        cleaned = []
        seen = set()
        for label in candidate_labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Candidate labels must be non-empty strings.")
            value = label.strip()
            key = value.casefold()
            if key in seen:
                raise ValueError(f"Duplicate candidate label: {value}")
            seen.add(key)
            cleaned.append(value)

        return text.strip(), cleaned

    def classify(
        self,
        text: str,
        candidate_labels: Optional[list[str]] = None,
        multi_label: bool = False,
    ) -> dict[str, Any]:
        passage, labels = self._validate(
            text, candidate_labels or DEFAULT_LABELS
        )
        model = self._load()
        tokenizer = model.tokenizer

        # Validate each premise-hypothesis pair before model inference.
        for label in labels:
            hypothesis = HYPOTHESIS_TEMPLATE.format(label)
            encoded = tokenizer(
                passage,
                hypothesis,
                truncation=False,
                add_special_tokens=True,
            )
            if len(encoded["input_ids"]) > tokenizer.model_max_length:
                raise ValueError(
                    f"Passage-label pair exceeds the model token limit for '{label}'."
                )

        start = perf_counter()
        raw = model(
            passage,
            candidate_labels=labels,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=multi_label,
        )
        elapsed = perf_counter() - start

        # Pipeline returns aligned label-score pairs. Keep them paired.
        ranked_labels = list(raw["labels"])
        ranked_scores = [float(score) for score in raw["scores"]]

        result = ClassificationResult(
            sequence=passage,
            labels=ranked_labels,
            scores=ranked_scores,
            model=self.model_id,
            revision=self.revision,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=multi_label,
            inference_seconds=elapsed,
        )
        return result.to_dict()
