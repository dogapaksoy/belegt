"""Combine a quote matcher with typed value checking."""

from __future__ import annotations

from collections.abc import Iterable

from belegt.matchers import FuzzyMatcher, QuoteMatcher
from belegt.normalize import normalize
from belegt.types import Claim, Reason, Span, ValueType, Verification
from belegt.values import value_in_text, value_in_text_approx


def _expand_to_words(document: str, span: Span) -> str:
    """Fuzzy alignments can clip the first/last characters; widen to whole tokens so a
    value at the edge ("…bis 15.03.2026.") is not cut off."""
    s, e = span.start, span.end
    while s > 0 and not document[s - 1].isspace():
        s -= 1
    while e < len(document) and not document[e].isspace():
        e += 1
    return document[s:e]


class Verifier:
    def __init__(
        self,
        matcher: QuoteMatcher | None = None,
        *,
        check_values: bool = True,
        ocr_tolerant: bool = True,
    ) -> None:
        self.matcher = matcher or FuzzyMatcher()
        self.check_values = check_values
        self.ocr_tolerant = ocr_tolerant

    @property
    def name(self) -> str:
        if not self.check_values:
            return self.matcher.name
        return self.matcher.name + ("+value~" if self.ocr_tolerant else "+value")

    def verify(self, document: str, claim: Claim) -> Verification:
        span = self.matcher.locate(document, claim.quote)
        if span is None:
            return Verification(claim, False, Reason.QUOTE_NOT_FOUND)
        if self.check_values and claim.type is not ValueType.TEXT:
            # Check the value against the *document* span, never against the model's quote:
            # the quote may itself be the hallucination.
            source, _ = normalize(_expand_to_words(document, span), casefold=False)
            found = value_in_text(claim.type, claim.value, source)
            if found is None:
                return Verification(claim, False, Reason.INVALID_VALUE, span)
            if not found:
                if self.ocr_tolerant and value_in_text_approx(claim.type, claim.value, source):
                    return Verification(claim, True, Reason.VALUE_APPROX, span)
                return Verification(claim, False, Reason.VALUE_NOT_IN_SOURCE, span)
        return Verification(claim, True, Reason.OK, span)

    def verify_all(self, document: str, claims: Iterable[Claim]) -> list[Verification]:
        return [self.verify(document, c) for c in claims]


def verify(
    document: str, claims: Iterable[Claim], *, threshold: float = 90.0
) -> list[Verification]:
    """Verify claims with the recommended default: fuzzy quote matching + value checks."""
    return Verifier(FuzzyMatcher(threshold)).verify_all(document, claims)
