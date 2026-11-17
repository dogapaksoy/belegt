import pytest

from belegt import Claim, ExactMatcher, FuzzyMatcher, NormalizedMatcher, Reason, ValueType, verify
from belegt.verifier import Verifier

DOC = """Finanzamt Karlsruhe

Karlsruhe, 02.03.2026

Steuernummer: 34/815/08150

Bitte überweisen Sie den Nachzahlungs-
betrag von 1.234,56 € bis spätestens
15.03.2026 auf das unten angegebene Konto.
Gegen diesen Bescheid können Sie innerhalb eines Monats Einspruch einlegen."""

PAY = ("Bitte überweisen Sie den Nachzahlungsbetrag von 1.234,56 € bis spätestens "
       "15.03.2026 auf das unten angegebene Konto.")  # fmt: skip

# The same letter after a bad scan.
NOISY = (DOC.replace("ü", "u").replace("m", "rn").replace("15.03.2026", "l5.O3.2026")
         .replace("Nachzahlungs", "Nachzahluugs"))  # fmt: skip


def deadline(value="2026-03-15", quote=PAY):
    return Claim("frist", value, quote, ValueType.DATE)


def test_exact_matching_fails_on_line_breaks_normalized_succeeds():
    assert ExactMatcher().locate(DOC, PAY) is None
    span = NormalizedMatcher().locate(DOC, PAY)
    assert span is not None
    assert span.text.startswith("Bitte überweisen") and span.text.endswith("Konto.")


def test_fuzzy_matching_survives_ocr_noise():
    assert NormalizedMatcher().locate(NOISY, PAY) is None
    span = FuzzyMatcher(85).locate(NOISY, PAY)
    assert span is not None and span.score >= 85


def test_correct_claims_supported_on_noisy_scan():
    claims = [
        deadline(),
        Claim("betrag", "1234.56", PAY, ValueType.AMOUNT),
        Claim("steuernummer", "34/815/08150", "Steuernummer: 34/815/08150", ValueType.ID),
    ]
    results = verify(NOISY, claims, threshold=85)
    assert [r.reason for r in results] == [Reason.OK] * 3


def test_right_quote_wrong_value_is_flagged():
    r = verify(DOC, [deadline("2026-03-16")])[0]
    assert not r.supported and r.reason is Reason.VALUE_NOT_IN_SOURCE
    # A quote-only verifier cannot catch this:
    assert (
        Verifier(FuzzyMatcher(), check_values=False).verify(DOC, deadline("2026-03-16")).supported
    )


def test_value_from_elsewhere_in_letter_is_flagged():
    # The letter date is in the document, but not in the cited sentence.
    r = verify(DOC, [deadline("2026-03-02")])[0]
    assert r.reason is Reason.VALUE_NOT_IN_SOURCE


def test_value_is_checked_against_document_not_against_quote():
    hallucinated_quote = PAY.replace("15.03.2026", "22.03.2026")
    r = verify(DOC, [deadline("2026-03-22", hallucinated_quote)])[0]
    assert not r.supported


def test_fabricated_quote_is_flagged():
    claim = Claim("todo", "Geburtsurkunde mitbringen", "Bitte bringen Sie Ihre Geburtsurkunde mit.")
    assert verify(DOC, [claim])[0].reason is Reason.QUOTE_NOT_FOUND


def test_invalid_claimed_value():
    assert verify(DOC, [deadline("bald")])[0].reason is Reason.INVALID_VALUE


@pytest.mark.parametrize("matcher", [ExactMatcher(), NormalizedMatcher(), FuzzyMatcher()])
def test_empty_quote_never_matches(matcher):
    assert matcher.locate(DOC, "   ") is None


def test_span_points_into_original_document():
    span = verify(NOISY, [deadline()], threshold=85)[0].span
    assert span is not None and NOISY[span.start : span.end] == span.text


def test_one_dropped_digit_is_accepted_but_marked_approx():
    doc = "Bitte zahlen Sie den Betrag bis zum 2.05.2026."  # OCR dropped the "9" of 29.05.
    quote = "Bitte zahlen Sie den Betrag bis zum 29.05.2026."
    r = verify(doc, [deadline("2026-05-29", quote)])[0]
    assert r.supported and r.reason is Reason.VALUE_APPROX
    strict = Verifier(FuzzyMatcher(), ocr_tolerant=False).verify(doc, deadline("2026-05-29", quote))
    assert strict.reason is Reason.VALUE_NOT_IN_SOURCE


def test_substituted_digit_is_never_tolerated():
    doc = "Bitte zahlen Sie den Betrag bis zum 29.05.2026."
    r = verify(doc, [deadline("2026-05-28", doc)])[0]
    assert r.reason is Reason.VALUE_NOT_IN_SOURCE


def test_value_hyphenated_across_lines():
    doc = "Bitte zahlen Sie den Betrag bis zum 25. Dez-\nember 2026."
    assert (
        verify(
            doc, [deadline("2026-12-25", "Bitte zahlen Sie den Betrag bis zum 25. Dezember 2026.")]
        )[0].reason
        is Reason.OK
    )


@pytest.mark.parametrize(
    "doc, claimed",
    [
        ("Der Betrag von 4.335,27 Euro ist fällig.", "43352.70"),  # x10 error
        ("Bitte zahlen Sie bis zum 26. September 2026.", "2026-09-09"),  # other day
        ("Bitte zahlen Sie bis zum 9. August 2026.", "2026-08-08"),
    ],
)
def test_tolerance_does_not_accept_hallucinated_values(doc, claimed):
    kind = ValueType.AMOUNT if "Betrag" in doc else ValueType.DATE
    r = verify(doc, [Claim("x", claimed, doc, kind)])[0]
    assert r.reason is Reason.VALUE_NOT_IN_SOURCE


def test_dropped_cent_digit_is_tolerated():
    doc = "ein Betrag von EUR 55,8 fällig."
    r = verify(
        doc, [Claim("betrag", "55.08", "ein Betrag von EUR 55,08 fällig.", ValueType.AMOUNT)]
    )
    assert r[0].reason is Reason.VALUE_APPROX
