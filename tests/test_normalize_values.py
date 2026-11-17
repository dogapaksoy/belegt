from datetime import date
from decimal import Decimal

import pytest

from belegt.normalize import normalize, to_original
from belegt.types import ValueType
from belegt.values import find_amounts, find_dates, parse_amount, repair_digits, value_in_text


def test_normalize_removes_layout_artifacts():
    text = "Bitte zahlen Sie die Zahlungs-\nfrist  „pünktlich“\n ein."
    norm, idx = normalize(text)
    assert norm == 'bitte zahlen sie die zahlungsfrist "pünktlich" ein.'
    s, e = to_original(idx, norm.index("zahlungsfrist"), norm.index("zahlungsfrist") + 13)
    assert text[s:e] == "Zahlungs-\nfrist"


def test_normalize_keeps_capitalized_hyphen_compounds():
    assert normalize("Kranken-\nKasse")[0] == "kranken- kasse"


def test_normalize_ligatures_and_decomposed_umlauts():
    assert normalize("Gebühr ﬁnden Straße")[0] == "gebühr finden strasse"


def test_repair_digits_only_touches_numeric_runs():
    assert repair_digits("Bis 15.O3.2O26 lS") == "Bis 15.03.2026 lS"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("bis zum 15.03.2026.", date(2026, 3, 15)),
        ("bis zum 5. März 2026", date(2026, 3, 5)),
        ("am 5.3.26 um", date(2026, 3, 5)),
        ("bis zum l5.O3.2O26", date(2026, 3, 15)),
    ],
)
def test_find_dates(text, expected):
    assert find_dates(text) == [expected]


def test_find_dates_rejects_impossible_dates():
    assert find_dates("am 31.02.2026") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Betrag von 1.234,56 €", Decimal("1234.56")),
        ("EUR 93,00 fällig", Decimal("93.00")),
        ("beträgt 100 Euro.", Decimal("100.00")),
        ("von 4.l89,36 EUR", Decimal("4189.36")),
    ],
)
def test_find_amounts(text, expected):
    assert find_amounts(text) == [expected]


def test_dates_and_years_are_not_amounts():
    assert find_amounts("am 15.03.2026 für das Jahr 2025") == []


@pytest.mark.parametrize("value", ["1234.56", "1.234,56", "1.234,56 €", "EUR 1234,56"])
def test_parse_amount_formats(value):
    assert parse_amount(value) == Decimal("1234.56")


def test_value_in_text_by_type():
    span = "Steuernummer: 34/815/O8150 – zahlen Sie 1.234,56 € bis 15. März 2026."
    assert value_in_text(ValueType.DATE, "2026-03-15", span)
    assert not value_in_text(ValueType.DATE, "2026-03-16", span)
    assert value_in_text(ValueType.AMOUNT, "1234.56", span)
    assert not value_in_text(ValueType.AMOUNT, "1243.56", span)
    assert value_in_text(ValueType.ID, "34/815/08150", span)
    assert not value_in_text(ValueType.ID, "34/815/08151", span)
    assert value_in_text(ValueType.DATE, "nächste Woche", span) is None


def test_ocr_damaged_month_names():
    assert find_dates("bis zum 23. Novemben 2026") == [date(2026, 11, 23)]
    assert find_dates("bis zum 3. Marz 2026") == [date(2026, 3, 3)]
    assert find_dates("am 3. Haus 2026") == []
