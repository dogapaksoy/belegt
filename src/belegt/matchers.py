"""Quote matchers: locate the quote a model cited inside the source document."""

from __future__ import annotations

from typing import Protocol

from rapidfuzz import fuzz

from belegt.normalize import normalize, to_original
from belegt.types import Span


class QuoteMatcher(Protocol):
    name: str

    def locate(self, document: str, quote: str) -> Span | None: ...


class ExactMatcher:
    """Baseline: the quote must appear verbatim."""

    name = "exact"

    def locate(self, document: str, quote: str) -> Span | None:
        q = quote.strip()
        i = document.find(q) if q else -1
        return None if i < 0 else Span(i, i + len(q), document[i : i + len(q)], 100.0)


class NormalizedMatcher:
    """Exact match after whitespace / hyphenation / quote / case normalization."""

    name = "normalized"

    def locate(self, document: str, quote: str) -> Span | None:
        nd, dmap = normalize(document)
        nq, _ = normalize(quote)
        i = nd.find(nq) if nq else -1
        if i < 0:
            return None
        s, e = to_original(dmap, i, i + len(nq))
        return Span(s, e, document[s:e], 100.0)


class FuzzyMatcher:
    """Best approximate alignment of the normalized quote in the normalized document.

    ``threshold`` is a rapidfuzz partial-ratio score (0-100). Lower values tolerate more
    OCR noise but also accept quotes that differ in meaning (e.g. an inserted "nicht").
    """

    def __init__(self, threshold: float = 90.0) -> None:
        self.threshold = threshold
        self.name = f"fuzzy@{threshold:g}"

    def locate(self, document: str, quote: str) -> Span | None:
        nd, dmap = normalize(document)
        nq, _ = normalize(quote)
        if not nq or not nd:
            return None
        i = nd.find(nq)
        if i >= 0:
            start, end, score = i, i + len(nq), 100.0
        elif len(nq) >= len(nd):  # e.g. a one-sentence document that lost characters to OCR
            start, end, score = 0, len(nd), fuzz.ratio(nq, nd)
            if score < self.threshold:
                return None
        else:
            # partial_ratio only scores windows exactly len(quote) long, which penalizes OCR
            # insertions/deletions ("m" -> "rn"). Use it to find a candidate region with a
            # relaxed cutoff, then refine the window boundaries with a full ratio.
            res = fuzz.partial_ratio_alignment(nq, nd, score_cutoff=max(0, self.threshold - 15))
            if res is None or res.dest_end <= res.dest_start:
                return None
            start, end, score = _refine(nq, nd, res.dest_start, res.dest_end)
            if score < self.threshold:
                return None
        s, e = to_original(dmap, start, end)
        return Span(s, e, document[s:e], score)


def _refine(quote: str, doc: str, start: int, end: int) -> tuple[int, int, float]:
    slack = max(3, len(quote) // 10)
    best = (start, end, fuzz.ratio(quote, doc[start:end]))
    for s in range(max(0, start - slack), start + slack + 1):
        for e in range(max(s + 1, end - slack), min(len(doc), end + slack) + 1):
            score = fuzz.ratio(quote, doc[s:e], score_cutoff=best[2])
            if score > best[2]:
                best = (s, e, score)
    return best
