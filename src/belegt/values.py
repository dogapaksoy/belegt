"""Typed value parsing for German documents, tolerant of common OCR digit errors.

A quote can be found in the document while the extracted value is still wrong
(the model cites the right sentence but copies the wrong date). So after locating
the quote we check that the claimed value actually occurs in the located span.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz, process

from belegt.types import ValueType

_MONTHS = {
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
    "november": 11, "dezember": 12,
}  # fmt: skip

# Characters OCR commonly produces in place of digits.
_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1", "i": "1", "|": "1",
     "!": "1", "Z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "b": "6", "g": "9"}
)  # fmt: skip
_DIGIT_RUN = re.compile(r"[0-9OoDQlIi|!ZSsBGbg]+")


def repair_digits(text: str) -> str:
    """Replace digit look-alikes inside runs that are mostly digits ("2O26" -> "2026").

    Words without digits are never touched, so "Bis" stays "Bis"."""

    def fix(m: re.Match[str]) -> str:
        run = m.group(0)
        digits = sum(c.isdigit() for c in run)
        if digits and len(run) >= 2 and digits * 2 >= len(run):
            return run.translate(_DIGIT_LOOKALIKES)
        return run

    return _DIGIT_RUN.sub(fix, text)


# ---------------------------------------------------------------- dates

_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})\.\s?(\d{1,2})\.\s?(\d{4}|\d{2})(?!\d)")
_TEXT_DATE = re.compile(r"(?<!\d)(\d{1,2})\.?\s*([^\W\d_]{3,10})\.?\s+(\d{4})(?!\d)")


def _month(word: str) -> int | None:
    """Month number for a possibly OCR-damaged German month name ("Novemben" -> 11)."""
    w = word.casefold()
    if w in _MONTHS:
        return _MONTHS[w]
    best = process.extractOne(w, _MONTHS.keys(), scorer=fuzz.ratio, score_cutoff=75)
    return _MONTHS[best[0]] if best else None


_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


def _mkdate(y: int, m: int, d: int) -> date | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list[date]:
    text = repair_digits(text)
    found: list[date | None] = []
    found += [_mkdate(int(y), int(m), int(d)) for d, m, y in _NUMERIC_DATE.findall(text)]
    for d, word, y in _TEXT_DATE.findall(text):
        m = _month(word)
        if m is not None:
            found.append(_mkdate(int(y), m, int(d)))
    found += [_mkdate(int(y), int(m), int(d)) for y, m, d in _ISO_DATE.findall(text)]
    return [d for d in found if d is not None]


def parse_date(value: str) -> date | None:
    dates = find_dates(value.strip())
    return dates[0] if len(dates) == 1 else None


# ---------------------------------------------------------------- amounts

_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}|-{1,2}))?(?![\d,]|\.\d)")
_CURRENCY_AFTER = re.compile(r"\s*(?:€|eur\b|euro\b)", re.IGNORECASE)
_CURRENCY_BEFORE = re.compile(r"(?:€|eur|euro)\s*$", re.IGNORECASE)


def _to_decimal(integer: str, cents: str | None) -> Decimal:
    cents = "00" if not cents or cents.startswith("-") else cents
    return Decimal(f"{integer.replace('.', '')}.{cents}")


def find_amounts(text: str) -> list[Decimal]:
    """Money amounts in German notation. A bare integer counts only next to a currency."""
    text = repair_digits(text)
    out = []
    for m in _NUMBER.finditer(text):
        has_cents = m.group(2) is not None
        has_currency = bool(_CURRENCY_AFTER.match(text, m.end())) or bool(
            _CURRENCY_BEFORE.search(text[max(0, m.start() - 6) : m.start()])
        )
        if has_cents or has_currency:
            out.append(_to_decimal(m.group(1), m.group(2)))
    return out


def parse_amount(value: str) -> Decimal | None:
    v = re.sub(r"(?i)€|eur(o)?", "", value).strip().replace(" ", "")
    try:
        if re.fullmatch(r"\d+(\.\d{1,2})?", v):  # "1234.56" — machine format
            return Decimal(v)
        if re.fullmatch(r"\d{1,3}(\.\d{3})*(,\d{1,2})?|\d+(,\d{1,2})?", v):  # German format
            return Decimal(v.replace(".", "").replace(",", "."))
    except InvalidOperation:
        return None
    return None


# ---------------------------------------------------------------- ids


def id_key(text: str) -> str:
    return re.sub(r"[\W_]", "", repair_digits(text).casefold())


# ---------------------------------------------------------------- dispatch


def value_in_text(kind: ValueType, value: str, text: str) -> bool | None:
    """Does ``value`` occur in ``text``? ``None`` means the claimed value itself is invalid."""
    if kind is ValueType.DATE:
        d = parse_date(value)
        return None if d is None else d in find_dates(text)
    if kind is ValueType.AMOUNT:
        a = parse_amount(value)
        return None if a is None else a in find_amounts(text)
    if kind is ValueType.ID:
        key = id_key(value)
        return None if not key else key in id_key(text)
    return True  # TEXT: the quote is the evidence


def _renderings(kind: ValueType, value: str) -> set[str]:
    """Keys (see :func:`id_key`) of the ways a date or ID may be printed in a letter."""
    if kind is ValueType.DATE:
        d = parse_date(value)
        if d is None:
            return set()
        month = [k for k, v in _MONTHS.items() if v == d.month][0]
        days, months = {f"{d.day}", f"{d:%d}"}, {f"{d.month}", f"{d:%m}", month}
        return {f"{dd}{mm}{d.year}" for dd in days for mm in months} | {f"{d:%d%m%y}"}
    if kind is ValueType.ID:
        return {id_key(value)}
    return set()


def _one_dropped(full: str, damaged: str) -> bool:
    return len(damaged) == len(full) - 1 and any(
        full[:i] + full[i + 1 :] == damaged for i in range(len(full))
    )


_AMOUNT_PARTS = re.compile(r"(?<![\d.,])(\d[\d.]*),(\d{1,2})(?![\d,])")


def value_in_text_approx(kind: ValueType, value: str, text: str) -> bool:
    """True if the value occurs in ``text`` with exactly one character dropped —
    the most common OCR error that strict parsing cannot recover ("29.05.2026" read as
    "2.05.2026"). Substitutions are *not* tolerated: "28.05." vs "29.05." is a
    different date, and exactly the kind of error an extractor hallucinates.

    Guards against accidental matches:
    * the damaged value must equal a whole run of 1-3 adjacent tokens that starts and
      ends with a digit-bearing token (so "September 2026" never stands in for a date);
    * amounts are compared part-wise, so a dropped digit cannot turn 4.335,27 into
      43.352,70 — euros or cents may lose one digit, never both, and never shift."""
    if kind is ValueType.AMOUNT:
        a = parse_amount(value)
        if a is None:
            return False
        euros, cents = f"{a:.2f}".split(".")
        for m in _AMOUNT_PARTS.finditer(repair_digits(text)):
            e, c = m.group(1).replace(".", ""), m.group(2)
            if (e == euros and _one_dropped(cents, c)) or (c == cents and _one_dropped(euros, e)):
                return True
        return False

    words = text.split()
    has_digit = [any(ch.isdigit() for ch in repair_digits(w)) for w in words]
    tokens = {
        id_key(" ".join(words[i:j]))
        for i in range(len(words))
        for j in range(i + 1, min(i + 4, len(words) + 1))
        if has_digit[i] and has_digit[j - 1]
    }
    return any(
        len(key) >= 5 and any(_one_dropped(key, t) for t in tokens)
        for key in _renderings(kind, value)
    )
