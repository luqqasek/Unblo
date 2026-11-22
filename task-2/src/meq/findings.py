"""Result cards for the control center: one small JSON file per check in research/findings/.

A card has a title, a few headline numbers ("tiles"), an optional table, optional rows for the dataset comparison
panel, and a pointer to a longer write-up.
"""
import json
import time

from . import config

DIR = config.ROOT / "research" / "findings"


def write(name: str, title: str, stage: str, tiles=(), table=None, compare=(), text=(), details: str | None = None,
          headline=()) -> None:
    """tiles: [{"label", "value", "hint"}]; table: {"columns": [...], "rows": [[...]]};
    compare: [{"dataset", "what", "family", "model", "score", "note"}]; details: path of a longer file, relative to the
    root; headline: tiles that also go in the top row of the control center."""
    DIR.mkdir(parents=True, exist_ok=True)
    card = {"name": name, "title": title, "stage": stage, "updated": time.strftime("%Y-%m-%d %H:%M"),
            "tiles": list(tiles), "table": table, "compare": list(compare), "text": list(text), "details": details,
            "headline": list(headline)}
    (DIR / f"{name}.json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")


def load_all() -> list[dict]:
    cards = []
    if DIR.exists():
        for path in sorted(DIR.glob("*.json")):
            try:
                cards.append(json.loads(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
    return cards
