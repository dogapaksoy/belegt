from belegt import bench
from belegt.cli import main
from belegt.data.synth import TEMPLATES, generate
from belegt.types import Claim
from belegt.verifier import verify


def test_generation_is_deterministic():
    assert generate(10, seed=3) == generate(10, seed=3)
    assert generate(10, seed=3) != generate(10, seed=4)


def test_every_letter_type_is_generated():
    assert {r["doc_type"] for r in generate(10)} == set(TEMPLATES)


def test_gold_claims_are_grounded_in_clean_documents():
    """Sanity check of the dataset itself: every gold claim must verify, verbatim."""
    for rec in generate(50, seed=1):
        for c in rec["claims"]:
            if c["label"] == "supported":
                assert c["quote"] in rec["document"], (rec["id"], c)
                assert verify(rec["document"], [Claim.from_dict(c)])[0].supported, (rec["id"], c)


def test_corrupted_claims_differ_from_gold():
    for rec in generate(50, seed=2):
        gold = {
            (c["field"], c["value"], c["quote"]) for c in rec["claims"] if c["corruption"] is None
        }
        for c in rec["claims"]:
            if c["corruption"]:
                assert (c["field"], c["value"], c["quote"]) not in gold


def test_bench_runs_and_value_checks_help():
    results = bench.run(generate(25), noise_levels=[0.0])
    by = {r.method: r for r in results}
    assert by["fuzzy@90+value"].f1 > by["fuzzy@90"].f1
    assert by["normalized"].false_flag_rate == 0.0
    assert "| noise |" in bench.to_markdown(results)


def test_cli_verify(tmp_path, capsys):
    doc = tmp_path / "letter.txt"
    doc.write_text("Bitte zahlen Sie den Betrag bis zum 15.03.2026.", encoding="utf-8")
    claims = tmp_path / "claims.json"
    claims.write_text(
        '[{"field": "frist", "value": "2026-03-15", "type": "date",'
        ' "quote": "Bitte zahlen Sie den Betrag bis zum 15.03.2026."}]',
        encoding="utf-8",
    )
    assert main(["verify", str(doc), str(claims), "--json"]) == 0
    assert '"supported": true' in capsys.readouterr().out
    assert main(["verify", str(doc), str(claims)]) == 0
    assert capsys.readouterr().out.startswith("✔ frist")
