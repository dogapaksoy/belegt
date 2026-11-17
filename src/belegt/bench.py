"""Benchmark verifiers on a labelled dataset across OCR noise levels.

Task framing: each claim is either supported by the document or not. A verifier
*flags* a claim when it returns ``supported=False``. Flagging an unsupported claim is a
true positive; flagging a correct claim is a false alarm (annoying for users, so we
report it separately as the false-flag rate).
"""

from __future__ import annotations

import statistics
import time
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from belegt.matchers import ExactMatcher, FuzzyMatcher, NormalizedMatcher
from belegt.noise import degrade
from belegt.types import Claim
from belegt.verifier import Verifier


def default_verifiers() -> list[Verifier]:
    return [
        Verifier(ExactMatcher(), check_values=False),
        Verifier(NormalizedMatcher(), check_values=False),
        Verifier(FuzzyMatcher(90), check_values=False),
        Verifier(NormalizedMatcher(), check_values=True, ocr_tolerant=False),
        Verifier(FuzzyMatcher(90), check_values=True, ocr_tolerant=False),
        Verifier(FuzzyMatcher(90), check_values=True, ocr_tolerant=True),
        Verifier(FuzzyMatcher(85), check_values=True, ocr_tolerant=True),
    ]


@dataclass
class Result:
    method: str
    noise: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    caught: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_flag_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    def recall_by(self, corruption: str) -> float:
        hit, total = self.caught[corruption]
        return hit / total if total else float("nan")

    def latency(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return (
            statistics.quantiles(self.latencies_ms, n=100)[int(q) - 1]
            if len(self.latencies_ms) > 1
            else self.latencies_ms[0]
        )


def run(
    records: Sequence[dict[str, Any]],
    verifiers: Iterable[Verifier] | None = None,
    noise_levels: Sequence[float] = (0.0, 0.03, 0.06, 0.1),
    seed: int = 0,
) -> list[Result]:
    verifiers = list(verifiers or default_verifiers())
    results = []
    for level in noise_levels:
        # Each document gets the same degradation for every verifier.
        docs = [degrade(r["document"], level, seed=seed ^ zlib.crc32(r["id"].encode()))
                for r in records]  # fmt: skip
        for v in verifiers:
            res = Result(v.name, level)
            for rec, doc in zip(records, docs, strict=True):
                for c in rec["claims"]:
                    claim = Claim.from_dict(c)
                    t0 = time.perf_counter()
                    flagged = not v.verify(doc, claim).supported
                    res.latencies_ms.append((time.perf_counter() - t0) * 1000)
                    if c["label"] == "unsupported":
                        res.caught[c["corruption"]][1] += 1
                        if flagged:
                            res.tp += 1
                            res.caught[c["corruption"]][0] += 1
                        else:
                            res.fn += 1
                    elif flagged:
                        res.fp += 1
                    else:
                        res.tn += 1
            results.append(res)
    return results


CORRUPTIONS = ["value_swap", "distractor", "cross_doc", "contradiction", "fabricated"]


def to_markdown(results: Sequence[Result]) -> str:
    head = (
        "| noise | method | F1 | precision | recall | false flags | "
        + " | ".join(f"recall: {c}" for c in CORRUPTIONS)
        + " | p95 ms |"
    )
    sep = "|" + "---|" * (7 + len(CORRUPTIONS))
    rows = [head, sep]
    for r in results:
        per = " | ".join(f"{r.recall_by(c):.2f}" for c in CORRUPTIONS)
        rows.append(
            f"| {r.noise:g} | `{r.method}` | **{r.f1:.3f}** | {r.precision:.3f} | {r.recall:.3f} "
            f"| {r.false_flag_rate:.1%} | {per} | {r.latency(95):.2f} |"
        )
    return "\n".join(rows)
