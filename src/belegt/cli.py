"""Command line: ``belegt verify``, ``belegt synth``, ``belegt bench``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from belegt import bench
from belegt.data.synth import generate, load_jsonl, save_jsonl
from belegt.noise import degrade
from belegt.types import Claim
from belegt.verifier import verify


def _cmd_verify(args: argparse.Namespace) -> int:
    document = Path(args.document).read_text(encoding="utf-8")
    raw = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    claims = [Claim.from_dict(c) for c in (raw["claims"] if isinstance(raw, dict) else raw)]
    results = verify(document, claims, threshold=args.threshold)
    if args.json:
        json.dump([r.to_dict() for r in results], sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        width = max(len(r.claim.field) for r in results) if results else 0
        for r in results:
            evidence = ""
            if r.span is not None:
                text = " ".join(r.span.text.split())
                evidence = repr(text if len(text) <= 50 else text[:49] + "…")
            mark = "✔" if r.supported else "✘"
            print(f"{mark} {r.claim.field:<{width}}  {r.reason.value:<19}  {evidence}".rstrip())
    return 0 if all(r.supported for r in results) else 1


def _cmd_synth(args: argparse.Namespace) -> int:
    records = generate(args.n, args.seed)
    save_jsonl(records, args.out)
    print(f"wrote {len(records)} letters, {sum(len(r['claims']) for r in records)} claims "
          f"to {args.out}")  # fmt: skip
    if args.show:
        print("\n" + degrade(records[0]["document"], args.show, seed=args.seed))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    records = list(load_jsonl(args.data)) if args.data else generate(args.n, args.seed)
    results = bench.run(records, noise_levels=args.noise, seed=args.seed)
    table = bench.to_markdown(results)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="belegt", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="verify extracted claims against a document")
    v.add_argument("document", help="UTF-8 text file (OCR / PDF text)")
    v.add_argument("claims", help='JSON list of {"field","value","quote","type"}')
    v.add_argument("--threshold", type=float, default=90.0)
    v.add_argument("--json", action="store_true", help="machine-readable output with offsets")
    v.set_defaults(func=_cmd_verify)

    s = sub.add_parser("synth", help="generate the synthetic benchmark")
    s.add_argument("-n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", default="belegt-synth.jsonl")
    s.add_argument("--show", type=float, metavar="NOISE", help="print one degraded example")
    s.set_defaults(func=_cmd_synth)

    b = sub.add_parser("bench", help="benchmark verifiers across noise levels")
    b.add_argument("--data", help="JSONL from `belegt synth` (default: generate on the fly)")
    b.add_argument("-n", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.1])
    b.add_argument("-o", "--out", help="also write the markdown table here")
    b.set_defaults(func=_cmd_bench)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
