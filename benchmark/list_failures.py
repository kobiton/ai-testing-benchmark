#!/usr/bin/env python3
"""Write a result file's failures, class by class, as a Markdown page someone can browse.

    python benchmark/list_failures.py benchmark-results/vision-qwen25-vl-7b-3phrasing-GT-v1-human-20260907-164510.json \
        --dataset data/dataset-v1-human.jsonl --source gpt --phrasing 0 --per-class 40 \
        > reference-results/failures/qwen2.5-vl-7b.md

One section per `answer_class` (the rule is `_answer_classes` in `artifacts/builder.py`).
Each row is the description the model was asked, what it did instead — for a wrong element,
the *description* of the element it chose, resolved from `--dataset` — and a link that opens
the viewer on exactly that element with the class filter set. Worst cases first. Made for
reading, not for counting: the counts are in the result file's `summary.answer_classes`.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.artifacts.builder import _answer_classes, gt_boxes_from_dataset, ANSWER_CLASSES  # noqa: E402

ORDER = ("other_element", "near_miss", "empty_space", "declined", "non_answer")
TITLES = {
    "other_element": ("Wrong element",
                      "The centre of the returned box lies inside a *different* labelled element on the same "
                      "screen: the model chose another control. \"Chose instead\" is that control's own description."),
    "near_miss": ("Near miss",
                  "The box overlaps the element asked for but its centre falls outside it: right control, loose "
                  "box. Kept apart from \"wrong element\" because calling it a wrong choice would be false. "
                  "IoU says how loose."),
    "empty_space": ("Empty space",
                    "The centre of the returned box lands on no labelled element at all."),
    "declined": ("Declined",
                 "No box: the model answered in prose. The only \"could not find\" this prompt allows, and rare, "
                 "because the prompt asks for a box and nothing else."),
    "non_answer": ("Non-answer", "Timeout, HTTP error or truncation: nothing the model decided."),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result")
    ap.add_argument("--dataset", default="", help="dataset JSONL, to name the element chosen instead and see every element on the screen")
    ap.add_argument("--source", default="", help="keep only element ids with this prefix (e.g. gpt)")
    ap.add_argument("--phrasing", type=int, default=None, help="description_index to keep (default: all)")
    ap.add_argument("--per-class", type=int, default=40, help="rows listed per class (0 = all)")
    ap.add_argument("--viewer", default="https://dataset-review-ui-test.kobiton.com", help="Result Analysis base URL for the links")
    ap.add_argument("--title", default="", help="model name for the heading (default: the file's `model`)")
    ap.add_argument("--label-boxes", action="store_true",
                    help="the rows are label boxes, not answers — say so in the header")
    a = ap.parse_args()

    d = json.load(open(a.result))
    rows = d.get("results") or []
    dataset = []
    if a.dataset:
        with open(a.dataset) as f:
            dataset = [json.loads(l) for l in f if l.strip()]
    describe = {(r.get("screenshot_id"), str(r.get("element_id"))): (r.get("descriptions") or [r.get("name", "")])[0]
                for r in dataset}
    for r in rows:  # the file's own rows name their element too
        describe.setdefault((r.get("screenshot_id"), str(r.get("element_id"))), r.get("description", ""))
    if not all("answer_class" in r for r in rows) or dataset:
        _answer_classes(rows, d.get("metric", "centroid"), gt_boxes_from_dataset(dataset) if dataset else None)
    if a.source:
        rows = [r for r in rows if str(r.get("element_id", "")).startswith(f"{a.source}-")]
    if a.phrasing is not None:
        rows = [r for r in rows if r.get("description_index", 0) == a.phrasing]

    by = defaultdict(list)
    for r in rows:
        by[r["answer_class"]].append(r)
    total = len(rows)
    name = Path(a.result).name
    title = a.title or d.get("model", "?")
    wrong = total - len(by.get("right_element", []))

    def link(r, cls):
        q = f"result={quote(name)}&screenshot={r['screenshot_id']}&element={quote(str(r['element_id']))}&class={cls}"
        if "description_index" in r:
            q += f"&phrasing={r['description_index']}"
        return f"{a.viewer}/compare?{q}"

    def md(s):
        return (s or "").replace("|", "\\|").replace("\n", " ")

    print(f"# {title} — the failures, one by one\n")
    what = ("**These rows are label boxes, not answers**: the boxes the model drew while describing each element "
            "when the corpus was built, compared with where the human annotator put the box. "
            if a.label_boxes else
            "Each row is one question the model was asked — a screenshot and one description — and what it did. ")
    print(what + f"Result file `{name}`" + (f", `{a.source}-` elements only" if a.source else "") +
          (f", phrasing {a.phrasing} (`name`)" if a.phrasing == 0 else f", phrasing {a.phrasing}" if a.phrasing is not None else "") +
          f": **{total:,}** answers, **{wrong:,}** of them wrong ({wrong / total:.1%}).\n" if total else "\n")
    print("Every answer is classified from the centre of its box against every human-drawn element on the same screen. "
          "The **open** link shows that screenshot in the viewer with the ground-truth box, the model's box, and — for a wrong "
          "element — the control it chose, drawn dashed red. Lists are worst-first and capped per class; the counts are complete.\n")
    print("| class | count | share |\n|---|---|---|")
    for c in ANSWER_CLASSES:
        n = len(by.get(c, []))
        print(f"| {TITLES.get(c, (c,))[0] if c != 'right_element' else 'Right element'} | {n:,} | {n / total:.1%} |" if total else f"| {c} | 0 | – |")
    print("\nTwo limits, by construction: the prompt allows no \"not found\" answer, so a model that cannot find the element "
          "has to guess and lands in one of the rows below; and every element here is present on its screen, so nothing "
          "measures how often a model invents a box for one that is not.\n")
    for c in ORDER:
        rs = by.get(c, [])
        if not rs:
            continue
        rs.sort(key=lambda r: (r.get("iou") or 0))
        shown = rs if not a.per_class else rs[:a.per_class]
        head, blurb = TITLES[c]
        print(f"\n## {head} — {len(rs):,}" + (f" (worst {len(shown)} shown)" if len(shown) < len(rs) else "") + f"\n\n{blurb}\n")
        if c == "other_element":
            print("| asked for | chose instead | open |\n|---|---|---|")
            for r in shown:
                tgt = describe.get((r["screenshot_id"], str(r.get("answer_class_target"))), "") or f"`{r.get('answer_class_target')}`"
                print(f"| {md(r.get('description'))} | {md(tgt)} | [open]({link(r, c)}) |")
        elif c == "near_miss":
            print("| asked for | IoU | open |\n|---|---|---|")
            for r in shown:
                print(f"| {md(r.get('description'))} | {(r.get('iou') or 0):.2f} | [open]({link(r, c)}) |")
        elif c == "declined":
            print("| asked for | model said | open |\n|---|---|---|")
            for r in shown:
                print(f"| {md(r.get('description'))} | {md((r.get('raw') or '')[:80])} | [open]({link(r, c)}) |")
        else:
            print("| asked for | open |\n|---|---|")
            for r in shown:
                print(f"| {md(r.get('description'))} | [open]({link(r, c)}) |")
    print(f"\n---\nGenerated by `benchmark/list_failures.py` from the result file named above; "
          f"re-run it with `--per-class 0` for every case.")


if __name__ == "__main__":
    main()
