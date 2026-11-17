# belegt

**Check that every fact an LLM extracts from a document is actually in the document — even when the document is a noisy scan.**

*belegt* (German: "evidenced, backed by proof") verifies structured LLM extractions — deadlines, amounts, reference numbers, obligations — against the source text. Each extracted value must cite a quote; *belegt* locates that quote in the document (robust to OCR errors and PDF layout) and checks that the claimed value really occurs there. Anything it cannot ground is flagged instead of shown to the user as fact.

It ships with a **benchmark of synthetic German administrative letters** (Finanzamt, Krankenkasse, Ausländerbehörde, Jobcenter, Rundfunkbeitrag) with labelled hallucination types and controllable OCR noise, so verification methods can be compared with numbers instead of anecdotes.

```
$ belegt verify examples/letter.txt examples/claims.json
✔ steuernummer  ok                   'Steuernumrner: 34/815/O8150'
✔ betrag        ok                   'Bitte überweisen Sie den Nachzahlungsbetrag von 1…'
✘ frist         value_not_in_source  'Bitte überweisen Sie den Nachzahlungsbetrag von 1…'
✘ einspruch     quote_not_found
```

The tax number is grounded despite OCR damage; the deadline is rejected because the model returned the letter date (02.03.2026), which is not in the cited sentence; the "Einspruch" sentence does not exist in this letter. Use `--json` for machine-readable output with character offsets.

```
```

## Why

LLM-based document assistants fail in a specific, dangerous way: the summary looks right, but one date is wrong. For an official letter, a wrong deadline is worse than no answer. Asking the model to cite its source helps only if someone *checks* the citation — and checking naively breaks on real input:

* **PDF/scan layout** — line breaks and hyphenation inside sentences (`Säumnis-\nzuschläge`) defeat verbatim matching even with perfect OCR. In the benchmark, exact matching falsely rejects **57 % of correct claims on clean, merely re-wrapped text**.
* **OCR errors** — `m→rn`, `0→O`, dropped characters, damaged month names (`Novemben`).
* **Right quote, wrong value** — the model cites the correct sentence but copies a different date from the letter. Quote-only checking cannot catch this at all (recall 0.00 on these cases).

## How it works

```
claim ─► normalize (layout, hyphenation, quotes, ligatures, case)  ─► locate quote
                                                                       exact → fuzzy alignment + boundary refinement
       ─► map span back to original offsets  ─► typed value check on the *document* span
                                                 date / amount / id, OCR digit repair,
                                                 optional one-dropped-character tolerance
       ─► optional NLI check for free-text claims: does the source sentence contradict the claim?
       ─► ok | value_approx | value_not_in_source | quote_not_found | contradicted | invalid_value
```

Design decisions worth noting:

* **The value is checked against the document span, never against the model's quote** — the quote itself may be the hallucination.
* **OCR tolerance is asymmetric.** A *dropped* digit (`29.05.2026` → `2.05.2026`) is accepted but reported as `value_approx` so a UI can ask for a quick manual check. A *substituted* digit (`28.05.` vs `29.05.`) is never tolerated: that is precisely what a hallucinated value looks like. Amounts are compared part-wise so a dropped digit cannot turn `4.335,27` into `43.352,70`.
* **Strings for form, a model for meaning.** Dates, amounts and IDs are checked exactly; a neural model is used only where exact checks are impossible — free-text claims such as obligations — and only after the quote has been located, so it judges one sentence pair instead of searching the whole document.
* Every result carries character offsets into the original text for highlighting the evidence.

## Usage

```python
from belegt import Claim, ValueType, verify

claims = [Claim(field="frist", type=ValueType.DATE, value="2026-03-15",
                quote="Bitte überweisen Sie den Betrag bis spätestens 15.03.2026.")]
for r in verify(ocr_text, claims):
    print(r.claim.field, r.supported, r.reason.value, r.span)
```

Semantic checking of free-text claims (adds PyTorch + a ~1 GB multilingual NLI model):

```python
from belegt.nli import NLIVerifier

verifier = NLIVerifier()          # fuzzy@85 + value checks + mDeBERTa NLI, contradiction ≥ 0.9
results = verifier.verify_all(ocr_text, claims)
```

```bash
pip install -e ".[dev]"            # or ".[dev,nli]" for the NLI verifier
belegt synth -n 200 -o data.jsonl --show 0.05   # generate benchmark, print a degraded letter
belegt bench --data data.jsonl [--nli]          # compare methods across noise levels
```

## Benchmark

200 letters, 2,120 claims (960 correct, 1,160 corrupted). Noise = per-character OCR confusion rate; every level, including 0, re-wraps and hyphenates the text like a scanned page. A *flag* on a corrupted claim is a true positive; **false flags** = correct claims wrongly rejected. Full table: [`results/synth-n200-seed0.md`](results/synth-n200-seed0.md).

| noise | method | F1 | false flags | recall: value swap | recall: contradiction |
|---|---|---|---|---|---|
| 0.00 | `exact` | 0.741 | 56.9 % | 0.48 | 1.00 |
| 0.00 | `fuzzy@90` (quote only) | 0.733 | 0.0 % | 0.00 | 0.88 |
| 0.00 | `fuzzy@90+value~` | 0.982 | 0.0 % | 1.00 | 0.88 |
| 0.00 | `fuzzy@85+value~+nli@0.9` | **1.000** | 0.0 % | 1.00 | **1.00** |
| 0.03 | `normalized+value` | 0.748 | 81.6 % | 1.00 | 1.00 |
| 0.03 | `fuzzy@90+value` (strict) | 0.946 | 9.8 % | 1.00 | 0.89 |
| 0.03 | `fuzzy@90+value~` | 0.978 | 1.7 % | 1.00 | 0.89 |
| 0.03 | `fuzzy@90+value~+nli@0.5` | 0.987 | 3.2 % | 1.00 | 1.00 |
| 0.03 | `fuzzy@85+value~+nli@0.9` | **0.992** | 1.7 % | 1.00 | **0.99** |
| 0.06 | `fuzzy@90+value~` | 0.957 | 7.6 % | 0.96 | 0.94 |
| 0.06 | `fuzzy@85+value~+nli@0.9` | **0.974** | 5.0 % | 0.96 | **0.99** |
| 0.10 | `fuzzy@90+value~` | 0.883 | 31.0 % | 0.98 | 0.99 |
| 0.10 | `fuzzy@85+value~` | 0.934 | 12.6 % | 0.98 | 0.90 |
| 0.10 | `fuzzy@85+value~+nli@0.9` | **0.943** | 13.4 % | 0.98 | **0.99** |

`value~` = value check with OCR tolerance; `nli@t` = [mDeBERTa-v3 multilingual NLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), reject if P(contradiction) ≥ t, applied to free-text claims only. Latency on CPU: <1 ms per claim for string methods, ~110 ms (p95) when the NLI model runs.

Findings:

1. **Value checking is the single biggest gain** (F1 0.73 → 0.98 on clean text). Quote matching alone misses every right-quote-wrong-value hallucination.
2. **OCR tolerance cuts false alarms ~6×** at moderate noise (9.8 % → 1.7 %) without losing recall.
3. **String similarity cannot detect contradictions.** A quote with an inserted negation scores >90 against the original, so 11–25 % of contradictions pass, and lowering the threshold for noisy scans makes it worse.
4. **NLI closes that gap — if calibrated.** At the default 0.5 threshold the model catches every contradiction but mistakes OCR-damaged correct sentences for contradictions (false flags 1.7 % → 3.2 %). Real contradictions score ≥ 0.98, so a 0.9 threshold keeps 99–100 % recall and removes most of those false alarms.
5. **The two methods complement each other.** With NLI guarding meaning, the string matcher can be loosened (90 → 85) to tolerate more OCR noise. The combination has the best F1 at every noise level.
6. **Heavy noise remains open.** At a 10 % character error rate, 13 % of correct claims are still flagged — mostly values destroyed beyond a single dropped character. This is where real OCR (instead of simulated noise) and confidence-aware thresholds come next.

## Roadmap

- [x] NLI verifier for `text` claims (contradiction detection), threshold calibration
- [ ] Compare German-specific NLI models (e.g. GBERT-based) and a distilled model for mobile latency
- [ ] LLM-as-judge baseline, with cost and latency
- [ ] End-to-end eval: real LLM extractors (open-weight vs. API) on the benchmark → hallucination rates per model
- [ ] Realistic OCR: render letters to images, run Tesseract / docTR instead of simulated noise
- [ ] Publish dataset on Hugging Face; PyPI release
- [ ] Adaptive threshold from an OCR-quality estimate

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest && ruff check src tests
```

MIT licensed. All letters, names and numbers in the dataset are synthetic.
