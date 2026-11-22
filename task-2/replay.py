"""Reconstruct submission 2411 from frozen evidence, without querying endpoints."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "research/experiments/s82"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "src"))
import v3
import v4
from meq.scoring import validate


def main():
    tree = ast.parse((SOURCE / "scored_v4.py").read_text())
    tags = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TAGS" for t in n.targets))
    saved = json.loads((SOURCE / "submissions/s82-05-v4.labels.json").read_text())
    predicted = {}
    for family in ("N", "Q", "L"):
        eps = sorted(e for e, (f, _) in saved.items() if f == family)
        # Later Qwen-3B evidence was not part of the accepted submission.
        lora = v3.lora_gains({"general": SOURCE / f"wb-ep-{family.lower()}.jsonl"})[0] if family != "N" else None
        quant = json.loads((SOURCE / f"qllr-{family.lower()}.json").read_text()) if family != "N" else None
        labels, _ = v4.label(eps, {e: set(tags.get(e, ())) for e in eps}, lora, quant, exclude_plain={"ep-26"})
        predicted.update({e: [family, labels[e]] for e in eps})
    assert predicted == saved, "Frozen reconstruction does not match accepted labels"
    artifact = ROOT / "submission/labels.csv"
    _, errors = validate(artifact)
    assert not errors, errors
    rows = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(artifact.open())}
    assert set(rows) == set(predicted) and len(rows) == 33
    for a in rows:
        for b in rows:
            assert (rows[a][0] == rows[b][0]) == (predicted[a][0] == predicted[b][0])
            assert (rows[a] == rows[b]) == (predicted[a] == predicted[b])
    receipt = json.loads((ROOT / "submission/receipt.json").read_text())
    assert hashlib.sha1(artifact.read_bytes()).hexdigest() == receipt["sha1"]
    print(json.dumps({"status": "PASS", "reconstructed_endpoints": 33,
                      "public_score": receipt["reply"]["score"], "submission_id": receipt["reply"]["submission_id"],
                      "hidden_score": None}, indent=2))


if __name__ == "__main__":
    main()
