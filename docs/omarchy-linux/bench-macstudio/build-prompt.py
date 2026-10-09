#!/usr/bin/env python3
"""Build the ~1357-token benchmark prompt (prompt.json).

Pads a fixed, deterministic filler text so the chat-formatted prompt
(system + user, generation prompt enabled) is ~1357 tokens under the
Qwen3.5 tokenizer shipped with mlx-community/Qwen3.5-0.8B-4bit.
Other engines' tokenizers count the same text differently; the receipt
records every engine's exact input token count.

Run on the Mac Studio:  $BENCH_ROOT (default ~/uzu-bench)/venv/bin/python build-prompt.py
"""
import json
import os
import sys

ROOT = os.environ.get("BENCH_ROOT", os.path.expanduser("~/uzu-bench"))
SNAP = f"{ROOT}/hf/hub/models--mlx-community--Qwen3.5-0.8B-4bit"
TARGET = 1357

SYSTEM = "You are a helpful assistant."
TASK = "Briefly summarize the text above."

SENTENCES = [
    "The assembly line runs on a schedule that shifts with the seasons.",
    "Buyers place orders weeks before the components arrive at the dock.",
    "A quotation that sits unanswered for five days is usually dead.",
    "Every factory keeps a private vocabulary for the same basic parts.",
    "The warehouse manager tracks temperature because resin degrades in heat.",
    "Forecasts fail first at the edges, where volumes are small and irregular.",
    "A single late shipment can erase the margin on an entire quarter.",
    "Distributors ask for exclusivity in regions they have never serviced.",
    "The best salespeople read drawings faster than the engineers do.",
    "Quality complaints travel upstream faster than purchase orders travel down.",
    "Tooling costs are amortized across runs that shrink every year.",
    "Customers compare lead times before they compare prices.",
    "The old catalog listed sizes that no modern machine can produce.",
    "Freight brokers solve problems that procurement never hears about.",
    "Samples travel by air; production travels by sea; patience travels by neither.",
    "A rejected lot returns with markings nobody can translate.",
    "The specification changed after the steel was already cut.",
    "Nobody signs the approval until somebody senior asks about price.",
    "Tradeshow leads go cold before the badge scanner is even synced.",
    "The plant adds a shift only when backlog survives three full weeks.",
    "Standard parts earn thin margins but keep the lights on.",
    "Engineers prototype in brass because it machines quickly.",
    "The quote template still references a fax number from 2004.",
    "Import duties depend on a classification that two consultants dispute.",
    "Survivors of every reorganization are the ones who know where the dies are stored.",
    "Customers reward candor about capacity with patience about schedules.",
    "The inventory system counts boxes; the floor counts pallets.",
    "A good rep knows which approvals are ceremonial and which are real.",
    "Retention is cheaper than acquisition in every industry that makes objects.",
    "The finishing line is the bottleneck nobody budgets to expand.",
    "Requests for expedited freight arrive most often on Friday afternoons.",
    "The archive room holds drawings for products discontinued decades ago.",
    "Inspection reports use adjectives where numbers would do.",
    "Small orders teach the shop discipline that large orders exploit.",
    "The newest machine runs oldest software because the license server is a beige tower.",
    "Every supplier claims ISO certification; audits tell different stories.",
    "Lead time promises are calibrated to the calendar, not the clock.",
    "Trade compliance reviews add a week nobody quotes to the customer.",
    "The scraper vendor promised integration and delivered a spreadsheet.",
    "Margins live in the gap between the quote and the invoice.",
]


def load_tokenizer():
    sys.path.insert(0, f"{ROOT}/venv/lib/python3.12/site-packages")
    from mlx_lm.utils import load

    # resolve snapshots dir
    snaps = [d for d in os.listdir(f"{SNAP}/snapshots") if not d.startswith(".")]
    path = f"{SNAP}/snapshots/{snaps[0]}"
    _, tok = load(path)
    return tok


def formatted_len(tok, user_text):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return len(ids)


def main():
    tok = load_tokenizer()
    # grow by repeating the pool until over target, then trim words
    text = " ".join(SENTENCES)
    while formatted_len(tok, f"{text}\n\n{TASK}") < TARGET:
        text += " " + " ".join(SENTENCES)
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if formatted_len(tok, f"{' '.join(words[:mid])}\n\n{TASK}") <= TARGET:
            lo = mid
        else:
            hi = mid - 1
    user_text = f"{' '.join(words[:lo])}\n\n{TASK}"

    final = formatted_len(tok, user_text)
    print(f"formatted prompt tokens: {final} (target {TARGET})")
    with open(f"{ROOT}/prompt.json", "w") as fh:
        json.dump({"system": SYSTEM, "user": user_text, "tokens_formatted_qwen35": final}, fh, indent=2)
    print(f"wrote {ROOT}/prompt.json")


if __name__ == "__main__":
    main()
