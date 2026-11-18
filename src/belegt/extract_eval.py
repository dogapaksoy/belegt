"""End-to-end evaluation: real LLMs extract facts from (noisy) letters, belegt checks them.

For every letter, the model receives the OCR-degraded text and a list of fields to
extract, each with a value and the verbatim quote it came from. We then measure:

* **extraction quality** — per gold field: correct / wrong / missing. A wrong value
  with a plausible quote is exactly the hallucination a user cannot spot.
* **verification** — how many wrong values belegt flags (catch rate) and how many
  correct values it flags by mistake (false alarms).
* **residual risk** — wrong values that are *not* flagged, i.e. errors a user would
  see presented as fact. This is the number that matters for a product.

Wrong values are split by cause. If OCR destroyed the gold value in the text the model
saw (e.g. "29.05.2026" scanned as "2.05.2026"), copying the damaged value is faithful
to the source — no text-level verifier can or should flag it. Only the remaining
**model errors** (hallucinated, mis-assigned or mis-converted values) are what a
grounding check is for, so catch rates are reported on those.

Only typed fields (dates, amounts, IDs) are scored, because their correctness can be
decided exactly; free-text obligations would need a second, fuzzier judge.

Requires the ``llm`` extra (torch + transformers). Generations are cached as JSONL so
long runs can be resumed and every raw output can be inspected.
"""

from __future__ import annotations

import json
import re
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from belegt.data.synth import generate
from belegt.matchers import FuzzyMatcher
from belegt.noise import degrade
from belegt.types import Claim, ValueType
from belegt.values import id_key, parse_amount, parse_date, value_in_text
from belegt.verifier import Verifier

FIELD_HELP: dict[str, str] = {
    "steuernummer": "the tax number (Steuernummer)",
    "versichertennummer": "the insurance number (Versichertennummer)",
    "aktenzeichen": "the case reference (Aktenzeichen)",
    "kundennummer": "the customer number (Kundennummer)",
    "beitragsnummer": "the contribution number (Beitragsnummer)",
    "betrag": "the amount the recipient has to pay",
    "gebuehr": "the processing fee",
    "frist": "the deadline by which the recipient must pay, submit or book something",
    "termin": "the date of the appointment the recipient is invited to",
    "ablaufdatum": "the date on which the residence permit expires",
}
TYPE_FORMAT = {
    ValueType.DATE: "date as YYYY-MM-DD",
    ValueType.AMOUNT: "number with a dot as decimal separator, e.g. 1234.56",
    ValueType.ID: "exactly as printed",
}

SYSTEM = "You extract facts from German official letters. You answer with JSON only."


def build_prompt(document: str, fields: Sequence[tuple[str, ValueType]]) -> str:
    lines = "\n".join(f"- {f} ({TYPE_FORMAT[t]}): {FIELD_HELP[f]}" for f, t in fields)
    return (
        "Extract the following fields from the letter below.\n\n"
        f"Fields:\n{lines}\n\n"
        'Return a JSON array of objects with the keys "field", "value" and "quote". '
        '"quote" must be the sentence or line of the letter that contains the value, '
        "copied exactly as it appears. Omit fields that are not in the letter.\n\n"
        f'Letter:\n"""\n{document}\n"""'
    )


def parse_output(text: str) -> list[dict[str, Any]]:
    """Best-effort JSON extraction from a model answer (code fences, chatter, ...)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):  # salvage individual objects
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                out.append(d)
        except json.JSONDecodeError:
            continue
    return out


def value_correct(kind: ValueType, predicted: str, gold: str) -> bool:
    if kind is ValueType.DATE:
        p = parse_date(predicted)
        return p is not None and p == parse_date(gold)
    if kind is ValueType.AMOUNT:
        p = parse_amount(predicted)
        return p is not None and p == parse_amount(gold)
    return bool(id_key(predicted)) and id_key(predicted) == id_key(gold)


# ------------------------------------------------------------------ generation


class HFChatModel:
    def __init__(self, model_name: str, max_new_tokens: int = 400) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,  # Qwen3 / SmolLM3: answer directly; ignored by others
        )
        with self._torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0, ids["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True)


def run_generation(
    model_name: str, n: int, noise: float, seed: int, cache: Path, verbose: bool = True
) -> list[dict[str, Any]]:
    """Generate (or resume) extractions for the first ``n`` letters; cached per model."""
    done: dict[str, dict[str, Any]] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    records = generate(max(n, 5), seed)[:n]
    todo = [r for r in records if r["id"] not in done]
    if todo:
        model = HFChatModel(model_name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(todo, 1):
                doc = degrade(rec["document"], noise, seed=seed ^ zlib.crc32(rec["id"].encode()))
                fields = [(c["field"], ValueType(c["type"])) for c in rec["claims"]
                          if c["corruption"] is None and c["type"] != "text"]  # fmt: skip
                t0 = time.perf_counter()
                raw = model.generate(build_prompt(doc, fields))
                row = {"id": rec["id"], "model": model_name, "noise": noise,
                       "seconds": round(time.perf_counter() - t0, 2), "output": raw}  # fmt: skip
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["id"]] = row
                if verbose:
                    print(
                        f"[{model_name}] {i}/{len(todo)} {rec['id']} {row['seconds']}s", flush=True
                    )
    return [done[r["id"]] for r in records]


# ------------------------------------------------------------------ scoring


@dataclass
class Score:
    model: str
    fields: int = 0
    correct: int = 0
    wrong: int = 0  # model errors: gold value was readable in the text the model saw
    ocr_destroyed: int = 0  # wrong only because OCR destroyed the gold value
    missing: int = 0
    wrong_flagged: int = 0
    correct_flagged: int = 0
    # correct value, but the cited quote does not contain it (or does not exist):
    # the flag is deserved, since the evidence shown to the user would be wrong
    correct_bad_citation: int = 0
    verbatim_quotes: int = 0
    quotes: int = 0
    seconds: list[float] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def pct(self, a: int, b: int) -> str:
        return f"{a / b:.1%}" if b else "–"

    @property
    def unflagged_errors(self) -> int:
        return self.wrong - self.wrong_flagged


def score(
    model_name: str,
    generations: Sequence[dict[str, Any]],
    noise: float,
    seed: int,
    verifier: Verifier | None = None,
) -> Score:
    verifier = verifier or Verifier(FuzzyMatcher(85))
    records = {r["id"]: r for r in generate(max(len(generations), 5), seed)}
    s = Score(model_name)
    for g in generations:
        rec = records[g["id"]]
        doc = degrade(rec["document"], noise, seed=seed ^ zlib.crc32(rec["id"].encode()))
        s.seconds.append(g["seconds"])
        predicted = {}
        for p in parse_output(g["output"]):
            name = str(p.get("field", "")).strip().lower()
            if name and name not in predicted and p.get("value") not in (None, ""):
                predicted[name] = p
        for c in rec["claims"]:
            if c["corruption"] is not None or c["type"] == "text":
                continue
            kind = ValueType(c["type"])
            s.fields += 1
            p = predicted.get(c["field"])
            if p is None:
                s.missing += 1
                continue
            claim = Claim(c["field"], str(p["value"]), str(p.get("quote") or ""), kind)
            s.quotes += 1
            s.verbatim_quotes += bool(claim.quote.strip()) and claim.quote.strip() in doc
            result = verifier.verify(doc, claim)
            flagged = not result.supported
            if value_correct(kind, claim.value, c["value"]):
                s.correct += 1
                s.correct_flagged += flagged
                if flagged and not (result.span and value_in_text(kind, claim.value, claim.quote)):
                    s.correct_bad_citation += 1
            elif not value_in_text(kind, c["value"], doc):
                s.ocr_destroyed += 1
            else:
                s.wrong += 1
                s.wrong_flagged += flagged
                if len(s.examples) < 20:
                    s.examples.append({"id": rec["id"], "field": c["field"], "gold": c["value"],
                                       "predicted": claim.value, "quote": claim.quote,
                                       "flagged": flagged})  # fmt: skip
    return s


def to_markdown(scores: Sequence[Score]) -> str:
    rows = [
        "| model | fields | correct | model errors | OCR-destroyed | missing | "
        "errors flagged | correct flagged (of which wrong citation) | **unflagged errors** "
        "| verbatim quotes | s / letter |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        sec = sorted(s.seconds)[len(s.seconds) // 2] if s.seconds else 0
        rows.append(
            f"| `{s.model.split('/')[-1]}` | {s.fields} | {s.pct(s.correct, s.fields)} "
            f"| {s.pct(s.wrong, s.fields)} | {s.pct(s.ocr_destroyed, s.fields)} "
            f"| {s.pct(s.missing, s.fields)} "
            f"| {s.pct(s.wrong_flagged, s.wrong)} "
            f"| {s.pct(s.correct_flagged, s.correct)} "
            f"({s.correct_bad_citation}/{s.correct_flagged}) "
            f"| **{s.pct(s.unflagged_errors, s.fields)}** "
            f"| {s.pct(s.verbatim_quotes, s.quotes)} | {sec:.0f} |"
        )
    return "\n".join(rows)
