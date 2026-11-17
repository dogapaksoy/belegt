"""Text normalization that keeps a map back to original character offsets.

PDF extraction and OCR change text in ways that break exact matching even when the
content is identical: line breaks inside sentences, words hyphenated across lines,
typographic quotes, ligatures, decomposed umlauts. We normalize both document and
quote the same way, match in normalized space, and map the match back to the
original document so the user can be shown the exact source span.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = {c: '"' for c in "„“”«»‟″"} | {c: "'" for c in "‚‘’‹›′`´"}
_DASHES = {c: "-" for c in "‐‑‒–—―−"}
_SOFT_HYPHEN = "­"

# "Zahlungs-\nfrist" -> "Zahlungsfrist": a hyphen at a line end followed by a lowercase
# continuation is a typesetting artifact, not part of the word.
_HYPHEN_BREAK = re.compile(r"(?<=\w)[-­][ \t]*\r?\n\s*(?=[a-zäöüß])")


def normalize(text: str, *, casefold: bool = True) -> tuple[str, list[int]]:
    """Return ``(normalized, index_map)`` where ``index_map[i]`` is the offset in
    ``text`` of the character that produced ``normalized[i]``."""
    dropped = bytearray(len(text))
    for m in _HYPHEN_BREAK.finditer(text):
        dropped[m.start() : m.end()] = b"\x01" * (m.end() - m.start())

    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if dropped[i] or ch == _SOFT_HYPHEN:
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
                prev_space = True
            continue
        if unicodedata.combining(ch) and out:
            # u + U+0308 -> ü ; keep the offset of the base character
            out[-1] = unicodedata.normalize("NFC", out[-1] + ch)
            continue
        ch = _QUOTES.get(ch) or _DASHES.get(ch) or ch
        piece = unicodedata.normalize("NFKC", ch)  # ligatures: "ﬁ" -> "fi"
        if casefold:
            piece = piece.casefold()  # also "ß" -> "ss"
        for p in piece:
            out.append(p)
            idx.append(i)
        prev_space = False

    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def to_original(index_map: list[int], start: int, end: int) -> tuple[int, int]:
    """Map a half-open span in normalized text back to the original text."""
    if not index_map or start >= end:
        raise ValueError("empty span")
    return index_map[start], index_map[end - 1] + 1
