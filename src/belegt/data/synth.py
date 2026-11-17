"""Synthetic German administrative letters with gold claims and labelled corruptions.

Real official letters contain personal data and cannot be published (GDPR), so the
benchmark is generated: realistic templates for common letter types, randomized
personal data, dates and amounts, and for every gold claim a *corrupted* claim of the
kind LLM extractors actually produce:

* ``value_swap``   – right quote, wrong value (date shifted, amount altered, ID digit changed)
* ``distractor``   – right quote, value copied from elsewhere in the letter (e.g. letter date)
* ``cross_doc``    – quote and value from another letter of the same type (near-identical text)
* ``contradiction``– quote rewritten to state the opposite (inserted/removed negation etc.)
* ``fabricated``   – plausible statement that does not appear in the letter at all
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from belegt.types import Claim, ValueType

MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
          "September", "Oktober", "November", "Dezember"]  # fmt: skip
FIRST = ["Doğa", "Lena", "Mehmet", "Sophie", "Jonas", "Ayşe", "Lukas", "Maria", "Elif",
         "Felix", "Anna", "Yusuf", "Paul", "Olena", "David", "Fatma", "Noah", "Clara"]  # fmt: skip
LAST = ["Schneider", "Yılmaz", "Müller", "Kowalski", "Weber", "Demir", "Fischer", "Nguyen",
        "Becker", "Hoffmann", "Kaya", "Schäfer", "Popescu", "Wagner", "Koch", "Şahin"]  # fmt: skip
CITIES = [("76131", "Karlsruhe"), ("69117", "Heidelberg"), ("10115", "Berlin"),
          ("80331", "München"),
          ("50667", "Köln"), ("70173", "Stuttgart"), ("68159", "Mannheim")]  # fmt: skip
STREETS = ["Kaiserstraße", "Hauptstraße", "Am Fasanengarten", "Lindenweg", "Bahnhofstraße",
           "Schillerplatz", "Goethestraße", "Rheinstraße"]  # fmt: skip
CLERKS = ["Frau Bauer", "Herr Krüger", "Frau Lehmann", "Herr Wolf", "Frau Zimmermann"]

FABRICATED_TEXT = [
    ("Bitte bringen Sie Ihre Geburtsurkunde im Original mit.", "Geburtsurkunde mitbringen"),
    ("Eine Ratenzahlung ist ohne weiteren Antrag möglich.", "Ratenzahlung ohne Antrag möglich"),
    ("Sie müssen zusätzlich eine Meldebescheinigung vorlegen.", "Meldebescheinigung vorlegen"),
    (
        "Der Bescheid ist bereits bestandskräftig und kann nicht angefochten werden.",
        "Bescheid nicht anfechtbar",
    ),
    ("Bitte senden Sie uns eine Kopie Ihres Arbeitsvertrags zu.", "Arbeitsvertrag senden"),
]
FABRICATED_DATED = [
    "Die persönliche Anhörung findet am {d} statt.",
    "Die nächste Rate wird am {d} automatisch abgebucht.",
    "Ihr Widerspruch muss bis zum {d} schriftlich vorliegen.",
]


# ------------------------------------------------------------------ formatting


def fmt_date(d: date, rng: random.Random) -> str:
    style = rng.random()
    if style < 0.5:
        return d.strftime("%d.%m.%Y")
    if style < 0.8:
        return f"{d.day}. {MONTHS[d.month - 1]} {d.year}"
    return f"{d.day}.{d.month}.{d.year}"


def fmt_amount(a: Decimal, rng: random.Random) -> str:
    integer, cents = f"{a:.2f}".split(".")
    grouped = f"{int(integer):,}".replace(",", ".")
    num = f"{grouped},{cents}"
    return rng.choice([f"{num} €", f"{num} EUR", f"EUR {num}", f"{num} Euro"])


def rand_amount(rng: random.Random, lo: int, hi: int) -> Decimal:
    return Decimal(rng.randint(lo * 100, hi * 100)) / 100


# ------------------------------------------------------------------ templates


@dataclass
class Gold:
    field: str
    type: ValueType
    value: str  # canonical: ISO date / "1234.56" / raw id / short paraphrase
    quote: str
    contradiction: str | None = None  # TEXT claims: a quote that states the opposite


@dataclass
class Letter:
    id: str
    doc_type: str
    document: str
    gold: list[Gold]
    dates: list[date] = field(default_factory=list)  # every date printed in the letter
    amounts: list[Decimal] = field(default_factory=list)


@dataclass
class _Ctx:
    rng: random.Random
    letter_date: date
    deadline: date
    first: str
    last: str
    ref: str = ""
    dates: list[date] = field(default_factory=list)
    amounts: list[Decimal] = field(default_factory=list)

    def d(self, value: date) -> str:
        self.dates.append(value)
        return fmt_date(value, self.rng)

    def a(self, value: Decimal) -> str:
        self.amounts.append(value)
        return fmt_amount(value, self.rng)


def _finanzamt(c: _Ctx) -> tuple[str, str, str, list[str], list[Gold]]:
    r = c.rng
    c.ref = f"{r.randint(10, 99)}/{r.randint(100, 999)}/{r.randint(10000, 99999)}"
    year = c.letter_date.year - 1
    amt = rand_amount(r, 80, 4800)
    pay = r.choice(
        [
            f"Bitte überweisen Sie den Nachzahlungsbetrag von {c.a(amt)} bis spätestens "
            f"{c.d(c.deadline)} auf das unten angegebene Konto.",
            f"Der Nachzahlungsbetrag in Höhe von {c.a(amt)} ist bis zum {c.d(c.deadline)} "
            "zu entrichten.",
        ]
    )
    einspruch = (
        "Gegen diesen Bescheid können Sie innerhalb eines Monats nach Bekanntgabe "
        "Einspruch einlegen."
    )
    zuschlag = "Bei verspäteter Zahlung werden Säumniszuschläge erhoben."
    paras = [
        f"für das Kalenderjahr {year} wird die Einkommensteuer wie folgt festgesetzt. "
        "Die festgesetzte Steuer übersteigt die geleisteten Vorauszahlungen. " + pay,
        f"{zuschlag} {einspruch}",
    ]
    gold = [
        Gold("steuernummer", ValueType.ID, c.ref, f"Steuernummer: {c.ref}"),
        Gold("betrag", ValueType.AMOUNT, f"{amt:.2f}", pay),
        Gold("frist", ValueType.DATE, c.deadline.isoformat(), pay),
        Gold(
            "einspruch",
            ValueType.TEXT,
            "Einspruch innerhalb eines Monats möglich",
            einspruch,
            "Gegen diesen Bescheid können Sie keinen Einspruch einlegen.",
        ),
        Gold(
            "folge",
            ValueType.TEXT,
            "Säumniszuschläge bei verspäteter Zahlung",
            zuschlag,
            "Bei verspäteter Zahlung werden keine Säumniszuschläge erhoben.",
        ),
    ]
    return ("Finanzamt", "Steuernummer", f"Bescheid für {year} über Einkommensteuer", paras, gold)


def _krankenkasse(c: _Ctx) -> tuple[str, str, str, list[str], list[Gold]]:
    r = c.rng
    c.ref = f"{r.choice('ABCDEFGHKLMNPRSTW')}{r.randint(100000000, 999999999)}"
    amt = rand_amount(r, 120, 2600)
    month = MONTHS[(c.letter_date.month - 2) % 12]
    rueck = (
        f"für den Zeitraum {month} {c.letter_date.year} besteht ein Beitragsrückstand "
        f"in Höhe von {c.a(amt)}."
    )
    frist = f"Bitte begleichen Sie den offenen Betrag bis zum {c.d(c.deadline)}."
    folge = (
        "Sollte die Zahlung nicht fristgerecht eingehen, sind wir gesetzlich verpflichtet, "
        "Säumniszuschläge zu erheben."
    )
    nachweis = "Bitte reichen Sie außerdem eine aktuelle Einkommensbescheinigung ein."
    paras = [f"{rueck} {frist}", folge, nachweis]
    gold = [
        Gold("versichertennummer", ValueType.ID, c.ref, f"Versichertennummer: {c.ref}"),
        Gold("betrag", ValueType.AMOUNT, f"{amt:.2f}", rueck),
        Gold("frist", ValueType.DATE, c.deadline.isoformat(), frist),
        Gold(
            "folge",
            ValueType.TEXT,
            "Säumniszuschläge bei verspäteter Zahlung",
            folge,
            "Sollte die Zahlung nicht fristgerecht eingehen, werden keine Säumniszuschläge "
            "erhoben.",
        ),
        Gold(
            "todo",
            ValueType.TEXT,
            "Einkommensbescheinigung einreichen",
            nachweis,
            "Bitte reichen Sie keine Einkommensbescheinigung ein.",
        ),
    ]
    return ("AOK Musterland", "Versichertennummer", "Beitragsrückstand", paras, gold)


def _auslaenderbehoerde(c: _Ctx) -> tuple[str, str, str, list[str], list[Gold]]:
    r = c.rng
    c.ref = f"ABH-{c.letter_date.year}-{r.randint(10000, 99999)}"
    expiry = c.deadline + timedelta(days=r.randint(20, 60))
    fee = Decimal(r.choice(["93.00", "98.00", "100.00", "62.00", "124.00"]))
    ablauf = f"Ihr Aufenthaltstitel läuft am {c.d(expiry)} ab."
    termin = r.choice(
        [
            f"Bitte vereinbaren Sie spätestens bis zum {c.d(c.deadline)} einen Termin zur "
            "Verlängerung.",
            f"Für die Verlängerung buchen Sie bitte bis zum {c.d(c.deadline)} einen Termin.",
        ]
    )
    unterlagen = (
        "Zum Termin bringen Sie bitte Ihren gültigen Reisepass, ein aktuelles "
        "biometrisches Lichtbild und einen Nachweis über Ihre Krankenversicherung mit."
    )
    gebuehr = f"Die Bearbeitungsgebühr beträgt {c.a(fee)}."
    paras = [f"{ablauf} {termin}", unterlagen, gebuehr]
    gold = [
        Gold("aktenzeichen", ValueType.ID, c.ref, f"Aktenzeichen: {c.ref}"),
        Gold("ablaufdatum", ValueType.DATE, expiry.isoformat(), ablauf),
        Gold("frist", ValueType.DATE, c.deadline.isoformat(), termin),
        Gold(
            "todo",
            ValueType.TEXT,
            "Reisepass, Lichtbild, Krankenversicherungsnachweis mitbringen",
            unterlagen,
            "Zum Termin müssen Sie weder Reisepass noch Lichtbild oder einen Nachweis über Ihre "
            "Krankenversicherung mitbringen.",
        ),
        Gold("gebuehr", ValueType.AMOUNT, f"{fee:.2f}", gebuehr),
    ]
    return (
        "Stadt – Ausländerbehörde",
        "Aktenzeichen",
        "Verlängerung Ihres Aufenthaltstitels",
        paras,
        gold,
    )


def _jobcenter(c: _Ctx) -> tuple[str, str, str, list[str], list[Gold]]:
    r = c.rng
    c.ref = f"{r.randint(100, 999)}D{r.randint(100000, 999999)}"
    appt = c.letter_date + timedelta(days=r.randint(7, 20))
    hour = r.choice(["08:30", "09:15", "10:00", "11:45", "13:30", "14:15"])
    einladung = f"Wir laden Sie zu einem persönlichen Gespräch am {c.d(appt)} um {hour} Uhr ein."
    unterlagen = (
        f"Bitte reichen Sie die fehlenden Kontoauszüge der letzten drei Monate bis zum "
        f"{c.d(c.deadline)} ein."
    )
    folge = (
        "Kommen Sie ohne wichtigen Grund nicht zu dem Termin, kann Ihr Bürgergeld gemindert werden."
    )
    paras = [einladung, unterlagen, folge]
    gold = [
        Gold("kundennummer", ValueType.ID, c.ref, f"Kundennummer: {c.ref}"),
        Gold("termin", ValueType.DATE, appt.isoformat(), einladung),
        Gold("frist", ValueType.DATE, c.deadline.isoformat(), unterlagen),
        Gold(
            "todo",
            ValueType.TEXT,
            "Kontoauszüge der letzten drei Monate einreichen",
            unterlagen,
            "Bitte reichen Sie keine Kontoauszüge der letzten drei Monate ein.",
        ),
        Gold(
            "folge",
            ValueType.TEXT,
            "Bürgergeld kann bei Nichterscheinen gemindert werden",
            folge,
            "Kommen Sie ohne wichtigen Grund nicht zu dem Termin, hat dies keine "
            "Auswirkungen auf Ihr Bürgergeld.",
        ),
    ]
    return ("Jobcenter", "Kundennummer", "Einladung zu einem Gespräch", paras, gold)


def _rundfunk(c: _Ctx) -> tuple[str, str, str, list[str], list[Gold]]:
    r = c.rng
    c.ref = f"{r.randint(100, 999)} {r.randint(100, 999)} {r.randint(100, 999)}"
    quarters = r.randint(1, 4)
    amt = Decimal("18.36") * 3 * quarters
    start = MONTHS[(c.letter_date.month - 1) % 12]
    faellig = (
        f"Für Ihre Wohnung ist für den Zeitraum ab {start} {c.letter_date.year} ein "
        f"Betrag von {c.a(amt)} fällig."
    )
    frist = f"Bitte zahlen Sie den Betrag bis zum {c.d(c.deadline)}."
    adresse = "Teilen Sie uns bitte mit, falls sich Ihre Anschrift geändert hat."
    paras = [f"{faellig} {frist}", adresse]
    gold = [
        Gold("beitragsnummer", ValueType.ID, c.ref, f"Beitragsnummer: {c.ref}"),
        Gold("betrag", ValueType.AMOUNT, f"{amt:.2f}", faellig),
        Gold("frist", ValueType.DATE, c.deadline.isoformat(), frist),
        Gold(
            "todo",
            ValueType.TEXT,
            "Adressänderung mitteilen",
            adresse,
            "Eine Änderung Ihrer Anschrift müssen Sie uns nicht mitteilen.",
        ),
    ]
    return ("Beitragsservice Rundfunk", "Beitragsnummer", "Zahlungsaufforderung", paras, gold)


TEMPLATES: dict[str, Callable[[_Ctx], tuple[str, str, str, list[str], list[Gold]]]] = {
    "finanzamt": _finanzamt,
    "krankenkasse": _krankenkasse,
    "auslaenderbehoerde": _auslaenderbehoerde,
    "jobcenter": _jobcenter,
    "rundfunk": _rundfunk,
}


def make_letter(doc_type: str, idx: int, rng: random.Random) -> Letter:
    letter_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 330))
    deadline = letter_date + timedelta(days=rng.randint(14, 35))
    c = _Ctx(rng, letter_date, deadline, rng.choice(FIRST), rng.choice(LAST))
    sender, ref_label, subject, paras, gold = TEMPLATES[doc_type](c)

    plz, city = rng.choice(CITIES)
    s_plz, s_city = rng.choice(CITIES)
    header_date = c.d(letter_date)
    doc = "\n".join([
        f"{sender} {s_city}",
        f"{rng.choice(STREETS)} {rng.randint(1, 120)} · {s_plz} {s_city}",
        "",
        f"{c.first} {c.last}",
        f"{rng.choice(STREETS)} {rng.randint(1, 120)}",
        f"{plz} {city}",
        "",
        f"{s_city}, {header_date}",
        "",
        f"{ref_label}: {c.ref}",
        "",
        f"Betreff: {subject}",
        "",
        f"Guten Tag {c.first} {c.last},",
        "",
        "\n\n".join(paras),
        "",
        "Mit freundlichen Grüßen",
        "Im Auftrag",
        rng.choice(CLERKS),
    ])  # fmt: skip
    return Letter(f"{doc_type}-{idx:04d}", doc_type, doc, gold, c.dates, c.amounts)


# ------------------------------------------------------------------ corruptions


def _shift_date(iso: str, rng: random.Random) -> str:
    d = date.fromisoformat(iso)
    delta = rng.choice([-1, 1]) * rng.choice([1, 2, 3, 7, 10, 14, 30])
    return (d + timedelta(days=delta)).isoformat()


def _alter_amount(value: str, rng: random.Random) -> str:
    a = Decimal(value)
    digits = list(f"{a:.2f}")
    pos = [i for i, ch in enumerate(digits) if ch.isdigit()]
    if rng.random() < 0.5 and len(pos) >= 2:  # transpose two adjacent digits, like a typo
        i = rng.randrange(len(pos) - 1)
        a_, b_ = pos[i], pos[i + 1]
        if digits[a_] != digits[b_]:
            digits[a_], digits[b_] = digits[b_], digits[a_]
            return "".join(digits)
    return f"{a * Decimal(rng.choice(['0.5', '1.1', '2', '10'])):.2f}"


def _alter_id(value: str, rng: random.Random) -> str:
    chars = list(value)
    pos = [i for i, ch in enumerate(chars) if ch.isdigit()]
    i = rng.choice(pos)
    chars[i] = rng.choice([d for d in "0123456789" if d != chars[i]])
    return "".join(chars)


def _corrupt(
    g: Gold, letter: Letter, siblings: list[Letter], rng: random.Random
) -> tuple[Claim, str]:
    if g.type is ValueType.TEXT:
        assert g.contradiction is not None
        return Claim(g.field, "NICHT: " + g.value, g.contradiction, g.type), "contradiction"

    options = ["value_swap", "cross_doc"]
    other_dates = [d for d in letter.dates if g.type is ValueType.DATE and d.isoformat() != g.value]
    other_amounts = [
        a for a in letter.amounts if g.type is ValueType.AMOUNT and f"{a:.2f}" != g.value
    ]
    if other_dates or other_amounts:
        options.append("distractor")
    kind = rng.choice(options)

    if kind == "distractor":
        v = (
            rng.choice(other_dates).isoformat()
            if other_dates
            else f"{rng.choice(other_amounts):.2f}"
        )
        return Claim(g.field, v, g.quote, g.type), kind
    if kind == "cross_doc":
        for other in rng.sample(siblings, len(siblings)):
            og = next((x for x in other.gold if x.field == g.field), None)
            if og and og.value != g.value:
                return Claim(g.field, og.value, og.quote, g.type), kind
    # value_swap (also the fallback when no usable sibling exists)
    alter = {ValueType.DATE: _shift_date, ValueType.AMOUNT: _alter_amount, ValueType.ID: _alter_id}
    return Claim(g.field, alter[g.type](g.value, rng), g.quote, g.type), "value_swap"


def _fabricate(letter: Letter, rng: random.Random) -> Claim:
    if rng.random() < 0.5:
        quote, value = rng.choice(FABRICATED_TEXT)
        return Claim("fabricated", value, quote, ValueType.TEXT)
    d = letter.dates[0] + timedelta(days=rng.randint(3, 40))
    quote = rng.choice(FABRICATED_DATED).format(d=fmt_date(d, rng))
    return Claim("fabricated", d.isoformat(), quote, ValueType.DATE)


# ------------------------------------------------------------------ dataset


def generate(n: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    """Generate ``n`` letters (round-robin over letter types) with labelled claims."""
    rng = random.Random(seed)
    types = list(TEMPLATES)
    letters = [make_letter(types[i % len(types)], i, rng) for i in range(n)]
    by_type: dict[str, list[Letter]] = {}
    for letter in letters:
        by_type.setdefault(letter.doc_type, []).append(letter)

    records = []
    for letter in letters:
        siblings = [x for x in by_type[letter.doc_type] if x is not letter]
        claims = []
        for g in letter.gold:
            pos = Claim(g.field, g.value, g.quote, g.type)
            claims.append({**pos.to_dict(), "label": "supported", "corruption": None})
            neg, kind = _corrupt(g, letter, siblings, rng)
            claims.append({**neg.to_dict(), "label": "unsupported", "corruption": kind})
        fab = _fabricate(letter, rng)
        claims.append({**fab.to_dict(), "label": "unsupported", "corruption": "fabricated"})
        records.append(
            {"id": letter.id, "doc_type": letter.doc_type, "document": letter.document,
             "claims": claims}
        )  # fmt: skip
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
