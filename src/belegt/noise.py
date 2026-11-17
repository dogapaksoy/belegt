"""Simulate what scanning a printed letter does to its text.

Two independent effects:
* layout: the text is re-wrapped at a page width, and long words are hyphenated at line
  ends. This alone breaks verbatim quote matching, even with perfect OCR.
* recognition errors: character confusions typical of OCR on German text
  (rn<->m, l<->1, 0<->O, ü->u, ß->B, ...), plus dropped characters.
"""

from __future__ import annotations

import random

_CONFUSIONS: dict[str, list[str]] = {
    "m": ["rn"], "r": ["n"], "n": ["u", "ri"], "u": ["n", "ü"], "l": ["1", "I"],
    "i": ["l", "í"], "1": ["l", "I"], "0": ["O", "o"], "O": ["0"], "o": ["0", "ö"],
    "ü": ["u", "ii"], "ä": ["a", "ä"], "ö": ["o"], "ß": ["B", "ss"], "e": ["c"],
    "c": ["e"], "h": ["b"], "5": ["S"], "8": ["B"], "6": ["b"], "f": ["t"], "t": ["f"],
}  # fmt: skip


def rewrap(text: str, width: int, hyphenate: float, rng: random.Random) -> str:
    paragraphs = text.split("\n\n")
    out = []
    for para in paragraphs:
        lines: list[str] = []
        for src_line in para.split("\n"):
            line = ""
            for word in src_line.split():
                if not line:
                    line = word
                elif len(line) + 1 + len(word) <= width:
                    line += " " + word
                else:
                    room = width - len(line) - 2
                    if len(word) >= 8 and room >= 3 and word.isalpha() and rng.random() < hyphenate:
                        cut = rng.randint(3, min(room, len(word) - 3))
                        lines.append(f"{line} {word[:cut]}-")
                        line = word[cut:]
                    else:
                        lines.append(line)
                        line = word
            if line:
                lines.append(line)
        out.append("\n".join(lines))
    return "\n\n".join(out)


def ocr_errors(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return text
    out = []
    for ch in text:
        r = rng.random()
        if ch in _CONFUSIONS and r < rate:
            out.append(rng.choice(_CONFUSIONS[ch]))
        elif ch.isalnum() and r < rate + rate / 6:
            continue  # dropped character
        else:
            out.append(ch)
    return "".join(out)


def degrade(text: str, level: float, seed: int = 0) -> str:
    """Return a scanned-looking version of ``text``. ``level`` is the per-character
    confusion rate (0.0 = layout changes only)."""
    rng = random.Random(seed)
    wrapped = rewrap(text, width=rng.randint(55, 80), hyphenate=0.6, rng=rng)
    return ocr_errors(wrapped, level, rng)
