"""NLIVerifier logic, tested with a stub model so CI needs no torch / model download."""

from belegt import Claim, Reason, ValueType
from belegt.nli import NLIVerifier

DOC = (
    "Gegen diesen Bescheid können Sie innerhalb eines Monats nach Bekanntgabe "
    "Einspruch einlegen. Bitte zahlen Sie bis zum 15.03.2026."
)
SOURCE = (
    "Gegen diesen Bescheid können Sie innerhalb eines Monats nach Bekanntgabe Einspruch einlegen."
)
NEGATED = SOURCE.replace("Einspruch", "keinen Einspruch")  # ~95 % string-similar to SOURCE


class StubNLI:
    """Says 'contradiction' whenever the hypothesis contains a negation word."""

    name = "stub/negation-detector"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.calls += pairs
        out = []
        for _, hyp in pairs:
            neg = any(w in hyp.split() for w in ("keinen", "keine", "nicht"))
            out.append(
                {"entailment": 0.05, "neutral": 0.05, "contradiction": 0.9}
                if neg
                else {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
            )
        return out


def test_negated_quote_passes_string_check_but_nli_rejects_it():
    stub = StubNLI()
    v = NLIVerifier(model=stub)  # type: ignore[arg-type]
    claim = Claim("einspruch", "kein Einspruch möglich", NEGATED)
    assert v.base.verify(DOC, claim).supported  # fuzzy match alone is fooled
    r = v.verify(DOC, claim)
    assert not r.supported and r.reason is Reason.CONTRADICTED
    premise, hypothesis = stub.calls[0]
    assert premise.startswith("Gegen diesen Bescheid") and hypothesis == NEGATED


def test_verbatim_quote_skips_the_model():
    stub = StubNLI()
    r = NLIVerifier(model=stub).verify(DOC, Claim("einspruch", "möglich", SOURCE))  # type: ignore[arg-type]
    assert r.supported and stub.calls == []


def test_typed_claims_are_left_to_the_value_check():
    stub = StubNLI()
    claim = Claim(
        "frist", "2026-03-15", "Bitte zahlen Sie nicht bis zum 15.03.2026.", ValueType.DATE
    )
    NLIVerifier(model=stub).verify(DOC, claim)  # type: ignore[arg-type]
    assert stub.calls == []


def test_claims_rejected_by_base_verifier_are_not_sent_to_model():
    stub = StubNLI()
    r = NLIVerifier(model=stub).verify(DOC, Claim("x", "y", "Ein völlig anderer Satz über Hunde."))  # type: ignore[arg-type]
    assert r.reason is Reason.QUOTE_NOT_FOUND and stub.calls == []


def test_entailment_mode_requires_support():
    class Unsure(StubNLI):
        def predict(self, pairs):
            return [{"entailment": 0.3, "neutral": 0.6, "contradiction": 0.1} for _ in pairs]

    claim = Claim("einspruch", "möglich", SOURCE.replace("Bekanntgabe", "Zustellung"))
    assert NLIVerifier(model=Unsure(), mode="contradiction").verify(DOC, claim).supported  # type: ignore[arg-type]
    assert not NLIVerifier(model=Unsure(), mode="entailment").verify(DOC, claim).supported  # type: ignore[arg-type]
