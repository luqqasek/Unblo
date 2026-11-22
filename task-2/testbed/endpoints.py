"""The AuditGym test bed: a second practice set with known answers, served on this Mac.

Every pseudo-endpoint is built with AuditGym's own endpoint code (external/auditgym/src, provider "local"), so the
disguises and variants are the ones its benchmark uses: hidden system prompts from its SystemChat list, decoding
changes, its four watermark families, and a published LoRA adapter at several strengths.

This is a separate validation check. The method under test only ever sees the text answers, exactly as with the
hackathon endpoints. Nothing here is used to identify or score the real endpoints.

Two "worlds" are supported because the real plain decoding is unknown until it is measured:
  deterministic  plain endpoints decode greedily (temperature 0)
  sampled        plain endpoints sample at temperature 0.7
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDITGYM = ROOT / "external" / "auditgym" / "src"

BASES = {
    "Q15": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
    "Q05": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "Q30": {"model": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1"},
    "OL1": {"model": "allenai/OLMo-2-0425-1B-Instruct", "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e"},
}
ADAPTERS = {   # published adapters and pins from AuditGym's benchmark/assets/trained-lora-v1.json; key = (family, name)
    ("Q15", "general"): {"adapter": "zjudai/flowertune-general-nlp-lora-qwen2.5-1.5b-instruct",
                         "revision": "acc92430c2106a406b1281d2c4b8a251970358ca", "name": "trained-qwen25_1_5b-general"},
    ("Q30", "persona"): {"adapter": "agastyasridharan/Qwen2.5-3B-Instruct-Sheldon-SFT-v2-LoRA",
                         "revision": "ada9409fe88fd7a8c35da0547fb07a632d8006c2", "name": "trained-qwen25_3b-persona"},
    ("Q30", "code"): {"adapter": "banty1614/codeforge-qwen-lora",
                      "revision": "3f0b298c030a49ee5d317f74d56ef70238556ee7", "name": "trained-qwen25_3b-code"},
    ("Q30", "latex"): {"adapter": "stefanj0/qwen2.5-3b-qlora-latex-fluentmath",
                       "revision": "b13f4638bdd91cc72a8ab4b95fbfdff772919fca", "name": "trained-qwen25_3b-latex"},
    ("Q30", "math"): {"adapter": "oscar128372/difl-qwen2.5-3b-math",
                      "revision": "00865a042be6c8a4e2c2c2f904d00bf2cec0f7e0", "name": "trained-qwen25_3b-math"},
}
WATERMARKS = {   # parameters from AuditGym's watermark studies
    "kgw": {"scheme": "kirchenbauer", "gamma": 0.25, "delta": 2.0},
    "synthid": {"scheme": "synthid", "ngram_len": 5, "depth": 30},
    "textseal": {"scheme": "textseal", "ngram": 3, "mixing_alpha": 0.9},
    "gumbel": {"scheme": "gumbel", "ngram": 3},
}
NEEDS_SAMPLING = {"synthid", "textseal", "gumbel"}     # these watermarks replace or reshape the sampler
PLAIN_TEMPERATURE = {"deterministic": 0.0, "sampled": 0.7}

# name, family, model label, kind, and the controls. "prompt" is an index into AuditGym's 15 SystemChat instructions
# (0-based: 2 first grader, 3 five words, 7 rhymes, 8 JSON, 10 Python tutor persona). "temperature" None = plain.
# "lora" is the adapter strength; "adapter" names which adapter of the family (default "general").
ENDPOINTS = [
    ("tb-01", "Q15", "base", "plain", {}),
    ("tb-02", "Q15", "base", "system prompt: exactly five words", {"prompt": 3}),
    ("tb-03", "Q15", "base", "persona: Python tutor", {"prompt": 10}),
    ("tb-04", "Q15", "base", "system prompt: rhymes", {"prompt": 7}),
    ("tb-05", "Q15", "base", "sampling at temperature 1.0", {"temperature": 1.0}),
    ("tb-06", "Q15", "base", "sampling at temperature 0.3", {"temperature": 0.3}),
    ("tb-07", "Q15", "base", "watermark: KGW", {"watermark": "kgw"}),
    ("tb-08", "Q15", "base", "watermark: SynthID-Text", {"watermark": "synthid"}),
    ("tb-09", "Q15", "base", "watermark: TextSeal", {"watermark": "textseal"}),
    ("tb-10", "Q15", "base", "watermark: Gumbel-max", {"watermark": "gumbel"}),
    ("tb-11", "Q15", "lora-0.3", "LoRA at strength 0.3", {"lora": 0.3}),
    ("tb-12", "Q15", "lora-0.7", "LoRA at strength 0.7", {"lora": 0.7}),
    ("tb-13", "Q15", "lora-1.0", "LoRA at strength 1.0", {"lora": 1.0}),
    ("tb-14", "Q15", "lora-1.5", "LoRA at strength 1.5", {"lora": 1.5}),
    ("tb-15", "Q15", "lora-0.7", "LoRA at strength 0.7 + system prompt: first grader", {"lora": 0.7, "prompt": 2}),
    ("tb-16", "Q15", "lora-1.5", "LoRA at strength 1.5 + sampling at temperature 1.0", {"lora": 1.5, "temperature": 1.0}),
    ("tb-17", "Q15", "lora-0.3", "LoRA at strength 0.3 + watermark: KGW", {"lora": 0.3, "watermark": "kgw"}),
    ("tb-18", "Q05", "base", "plain", {}),
    ("tb-19", "Q05", "base", "system prompt: JSON objects", {"prompt": 8}),
    ("tb-20", "Q05", "base", "sampling at temperature 1.0", {"temperature": 1.0}),
    ("tb-21", "Q05", "base", "watermark: KGW", {"watermark": "kgw"}),
    # added 2026-11-21: Qwen2.5-3B with four published adapters, among them a persona adapter
    ("tb-22", "Q30", "base", "plain", {}),
    ("tb-23", "Q30", "base", "system prompt: exactly five words", {"prompt": 3}),
    ("tb-24", "Q30", "base", "persona: Python tutor", {"prompt": 10}),
    ("tb-25", "Q30", "base", "sampling at temperature 1.0", {"temperature": 1.0}),
    ("tb-26", "Q30", "base", "watermark: KGW", {"watermark": "kgw"}),
    ("tb-27", "Q30", "persona-1.0", "persona LoRA at strength 1.0", {"adapter": "persona", "lora": 1.0}),
    ("tb-28", "Q30", "persona-0.5", "persona LoRA at strength 0.5", {"adapter": "persona", "lora": 0.5}),
    ("tb-29", "Q30", "code-1.0", "code LoRA at strength 1.0", {"adapter": "code", "lora": 1.0}),
    ("tb-30", "Q30", "latex-1.0", "LaTeX maths LoRA at strength 1.0", {"adapter": "latex", "lora": 1.0}),
    ("tb-31", "Q30", "math-1.0", "maths LoRA at strength 1.0", {"adapter": "math", "lora": 1.0}),
    ("tb-32", "Q30", "persona-1.0", "persona LoRA at strength 1.0 + system prompt: exactly five words", {"adapter": "persona", "lora": 1.0, "prompt": 3}),
    ("tb-33", "Q30", "code-1.0", "code LoRA at strength 1.0 + sampling at temperature 1.0", {"adapter": "code", "lora": 1.0, "temperature": 1.0}),
    # added 2026-11-21: OLMo-2 1B (the task text names OLMo); disguises only
    ("tb-34", "OL1", "base", "plain", {}),
    ("tb-35", "OL1", "base", "system prompt: rhymes", {"prompt": 7}),
    ("tb-36", "OL1", "base", "persona: Python tutor", {"prompt": 10}),
    ("tb-37", "OL1", "base", "sampling at temperature 1.0", {"temperature": 1.0}),
    ("tb-38", "OL1", "base", "sampling at temperature 0.3", {"temperature": 0.3}),
    ("tb-39", "OL1", "base", "watermark: KGW", {"watermark": "kgw"}),
    ("tb-40", "OL1", "base", "watermark: Gumbel-max", {"watermark": "gumbel"}),
    ("tb-41", "OL1", "base", "watermark: SynthID-Text", {"watermark": "synthid"}),
]


@lru_cache(maxsize=None)
def system_prompts() -> tuple:
    """AuditGym's fifteen SystemChat instructions, in file order."""
    data = json.loads((AUDITGYM / "benchmark" / "systemchat-v2.json").read_text(encoding="utf-8"))
    found = []

    def walk(node):
        if isinstance(node, dict):
            if {"mode", "content", "sha256"} <= set(node) and node["content"] not in found:
                found.append(node["content"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return tuple(found)


@lru_cache(maxsize=None)
def chat_template_spec(family: str) -> dict:
    """AuditGym only applies a hidden system prompt when the tokenizer's chat template is pinned by hash."""
    from transformers import AutoTokenizer

    from endpoint.chat_templates import DEFAULT_TEMPLATE_ID, content_sha256
    base = BASES[family]
    tokenizer = AutoTokenizer.from_pretrained(base["model"], revision=base["revision"])
    return {"scope": "default", "template": DEFAULT_TEMPLATE_ID, "sha256": content_sha256(tokenizer.chat_template),
            "tokenizer": base["model"], "tokenizer_revision": base["revision"]}


def build(name: str, world: str = "deterministic") -> tuple[dict, float]:
    """(AuditGym endpoint config, request temperature) for one pseudo-endpoint."""
    _, family, _, _, controls = next(row for row in ENDPOINTS if row[0] == name)
    config = {**BASES[family], "provider": "local"}
    if "prompt" in controls:
        text = system_prompts()[controls["prompt"]]
        config["system_prompt"] = {"mode": "prepend", "content": text,
                                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        config["chat_template"] = chat_template_spec(family)
    if "lora" in controls:
        adapter = ADAPTERS[(family, controls.get("adapter", "general"))]
        config["lora"] = {**adapter, "base_model": BASES[family]["model"], "base_revision": BASES[family]["revision"],
                          "scale": controls["lora"]}
    temperature = controls.get("temperature", PLAIN_TEMPERATURE[world])
    if "watermark" in controls:
        scheme = controls["watermark"]
        config["watermark"] = {**WATERMARKS[scheme], "key": 42, "key_ref": f"public://testbed/{scheme}-key-42"}
        if scheme in NEEDS_SAMPLING and temperature == 0:
            temperature = 1.0
    return config, temperature


def truth_rows() -> list[tuple]:
    """(endpoint, family, model, kind): the same columns as the organizers' practice_answers.csv."""
    return [(name, family, f"{family}-{model}", kind) for name, family, model, kind, _ in ENDPOINTS]
