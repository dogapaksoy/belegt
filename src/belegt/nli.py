"""Semantic verification with a natural-language-inference (NLI) model.

String similarity cannot tell "Sie können Einspruch einlegen" from "Sie können keinen
Einspruch einlegen" — the strings are ~95 % similar. An NLI model can: given the
sentence that is actually in the document (premise) and the sentence the extractor
claims (hypothesis), it predicts entailment / neutral / contradiction.

Requires the optional dependencies: ``pip install "belegt[nli]"``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from belegt.matchers import FuzzyMatcher
from belegt.normalize import normalize
from belegt.types import Claim, Reason, ValueType, Verification
from belegt.verifier import Verifier, expand_to_words

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class NLIModel:
    """Thin wrapper around a Hugging Face sequence-classification NLI model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only without extras
            raise ImportError('NLI verification needs: pip install "belegt[nli]"') from e
        self._torch = torch
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        labels = {i: str(lbl).lower() for i, lbl in self.model.config.id2label.items()}
        self.index = {name: i for i, name in labels.items()}
        missing = {"entailment", "contradiction"} - set(self.index)
        if missing:
            raise ValueError(f"{model_name} has labels {labels}; need entailment/contradiction")

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
        """Return label probabilities for each (premise, hypothesis) pair."""
        out: list[dict[str, float]] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            enc = self.tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                probs = self.model(**enc).logits.softmax(-1).cpu().tolist()
            out += [{name: row[j] for name, j in self.index.items()} for row in probs]
        return out


class NLIVerifier:
    """Run the string/value verifier first; for claims it accepts, ask an NLI model
    whether the document sentence supports the claimed sentence.

    ``mode="contradiction"`` rejects only when the model is confident the source says
    the opposite (few false alarms). ``mode="entailment"`` additionally requires the
    source to entail the claim (stricter; also catches unsupported additions).
    By default only free-text claims are checked: dates, amounts and IDs are already
    verified exactly by the value check.
    """

    def __init__(
        self,
        base: Verifier | None = None,
        model: NLIModel | str = DEFAULT_MODEL,
        *,
        mode: Literal["contradiction", "entailment"] = "contradiction",
        threshold: float = 0.9,
        types: Iterable[ValueType] = (ValueType.TEXT,),
    ) -> None:
        # The NLI step catches the meaning changes a loose string threshold lets through,
        # so the base matcher can afford to be more tolerant of OCR noise (benchmark:
        # fuzzy@85 + NLI@0.9 has the best F1 at every noise level).
        self.base = base or Verifier(FuzzyMatcher(85))
        self.nli = NLIModel(model) if isinstance(model, str) else model
        self.mode = mode
        self.threshold = threshold
        self.types = frozenset(types)

    @property
    def name(self) -> str:
        short = self.nli.name.rsplit("/", 1)[-1].split("-")[0]
        tag = "contra" if self.mode == "contradiction" else "entail"
        return f"{self.base.name}+nli:{short}:{tag}@{self.threshold:g}"

    def _rejects(self, probs: dict[str, float]) -> bool:
        if self.mode == "contradiction":
            return probs["contradiction"] >= self.threshold
        return probs["entailment"] < self.threshold

    def verify_all(self, document: str, claims: Iterable[Claim]) -> list[Verification]:
        results = self.base.verify_all(document, claims)
        todo, pairs = [], []
        for i, r in enumerate(results):
            if not (r.supported and r.span is not None and r.claim.type in self.types):
                continue
            premise, _ = normalize(expand_to_words(document, r.span), casefold=False)
            if normalize(premise)[0] == normalize(r.claim.quote)[0]:
                continue  # verbatim quote: nothing for the model to judge
            todo.append(i)
            pairs.append((premise, r.claim.quote))
        if not todo:
            return results
        for i, probs in zip(todo, self.nli.predict(pairs), strict=True):
            if self._rejects(probs):
                r = results[i]
                results[i] = Verification(r.claim, False, Reason.CONTRADICTED, r.span)
        return results

    def verify(self, document: str, claim: Claim) -> Verification:
        return self.verify_all(document, [claim])[0]

    def describe(self) -> dict[str, Any]:
        return {"model": self.nli.name, "mode": self.mode, "threshold": self.threshold}
