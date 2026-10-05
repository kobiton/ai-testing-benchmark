#!/usr/bin/env python3
"""Score the tree-then-screenshot cascade from an `xml-…` and a `vision-…` result file.

    python score_cascade.py benchmark-results/xml-gpt-5.6-terra-….json \\
                            benchmark-results/vision-gpt-5.6-terra-….json [XML2 VISION2 …] \\
                            [--source gpt] [--json cascade.json] [--with-vision-alone]

One Markdown block per pair: the eight cascade figures, overall and per phrasing, with the
sub-lines of "not answered" and the footnotes a reader needs to weigh them (keyboard rows, the
vision prompt's ability to decline, a cross-model pair). The rules are `artifacts/cascade.py`;
this file only renders them. No model is called and no image is read, so the whole corpus
scores in seconds and the pass can be re-run whenever one of the two files is re-scored.

The vision file need only cover the rows XML did not answer, so a run made with `--only-from`
and a full vision run print the same block; see `artifacts/cascade.py`. A `--json` file holds
the same numbers, one object per pair, for the README table and the result viewer.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.artifacts.cascade import ITEMS, XML_SUBLINES, VISION_SUBLINES, join_cascade  # noqa: E402

SUBLINE_TITLES = {
    "not_found": "`NOT_FOUND`",
    "invalid_xpath": "XPath did not compile",
    "no_match": "XPath matched no node",
    "declined": "declined (no box)",
    "non_answer": "no usable answer (parse failure, error)",
}


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _cell(block: dict, key: str) -> str:
    n = block["counts"][key]
    return f"{n} ({block['shares'][key] * 100:.1f}%)"


def _sub_cell(block: dict, group: str, key: str) -> str:
    n = block[group][key]
    total = block["total"]
    return f"{n} ({n / total * 100:.1f}%)" if total else "0"


def render(pair_name: str, xml: dict, vision: dict, res: dict, xml_path: str, vision_path: str,
           with_vision_alone: bool = False) -> str:
    blocks = [("all", res["overall"])] + [(f"{idx} · {b['style']}", b) for idx, b in res["by_description"].items()]
    cols = [name for name, _ in blocks]
    lines = [f"## {pair_name}", ""]
    prompt = f" · prompt `{xml['xml_prompt']}` ({xml.get('xml_prompt_sha256', '')})" if xml.get("xml_prompt") else ""
    lines.append(f"- XML: `{xml.get('model')}`{prompt} · `{Path(xml_path).name}`")
    lines.append(f"- Vision: `{vision.get('model')}` · prompt style `{vision.get('prompt_style', '')}` · `{Path(vision_path).name}`")
    o = res["overall"]
    lines.append(f"- Population: {o['total']} rows; vision consulted on the {o['vision_rows_used']} rows XML did not answer. "
                 f"Excluded: {res['excluded_xml_errors']} XML transport errors, "
                 f"{res['excluded_fall_through_no_vision_row']} fall-through rows with no vision row.")
    lines.append("")
    lines.append("| # | Outcome | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in cols) + "|")
    for i, (key, title) in enumerate(ITEMS, start=1):
        lines.append(f"| {i} | {title} | " + " | ".join(_cell(b, key) for _, b in blocks) + " |")
        if key == "xml_not_answered":
            for sub in XML_SUBLINES:
                lines.append(f"|   | &nbsp;&nbsp;of which {SUBLINE_TITLES[sub]} | "
                             + " | ".join(_sub_cell(b, "xml_not_answered_by", sub) for _, b in blocks) + " |")
        if key == "vision_not_answered":
            for sub in VISION_SUBLINES:
                lines.append(f"|   | &nbsp;&nbsp;of which {SUBLINE_TITLES[sub]} | "
                             + " | ".join(_sub_cell(b, "vision_not_answered_by", sub) for _, b in blocks) + " |")
    if with_vision_alone and res["vision_covers_all_rows"]:
        lines.append("| | Vision alone on the same rows, correct | "
                     + " | ".join(f"{b['vision_alone_correct']} ({b['vision_alone_accuracy'] * 100:.1f}%)" for _, b in blocks) + " |")
    lines.append("")

    notes = []
    if with_vision_alone and not res["vision_covers_all_rows"]:
        notes.append("Vision alone is not available: the vision file covers only part of the population.")
    if o["keyboard_rows"]:
        notes.append(f"{o['keyboard_rows']} rows describe a key on the on-screen keyboard; "
                     f"{o['keyboard_rows_in_fall_through']} of them fell through to vision. The XML dumps hold the application window only, so a keyboard key cannot be found in the tree here.")
    if vision.get("prompt_style") != "production":
        notes.append(f"The vision file's prompt (`{vision.get('prompt_style', '')}`) does not let the model decline, so item 6 counts only unparseable or failed answers, not a \"not found\".")
    if xml.get("model") != vision.get("model"):
        notes.append("The two halves come from different models; a per-model cascade needs both files from the same one.")
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", nargs="+", metavar="FILE", help="XML result, vision result, repeated per pair")
    ap.add_argument("--source", default="", help="keep only element ids with this prefix (e.g. gpt), as summary.by_source does")
    ap.add_argument("--json", default="", help="also write the numbers here, one object per pair")
    ap.add_argument("--with-vision-alone", action="store_true",
                    help="add the screenshot strategy's own figure on the same rows (needs a full vision run)")
    a = ap.parse_args()
    if len(a.pairs) % 2:
        ap.error("files come in pairs: XML result then vision result")

    out = []
    for xml_path, vision_path in zip(a.pairs[0::2], a.pairs[1::2]):
        xml, vision = _load(xml_path), _load(vision_path)
        if xml.get("input") != "xml":
            ap.error(f"{xml_path} is not an --input xml result")
        if vision.get("input", "screenshot") != "screenshot":
            ap.error(f"{vision_path} is not a vision result")
        res = join_cascade(xml.get("results") or [], vision.get("results") or [], source=a.source)
        name = f"{xml.get('model')} (XML) → {vision.get('model')} (vision)"
        print(render(name, xml, vision, res, xml_path, vision_path, with_vision_alone=a.with_vision_alone))
        out.append({"xml_file": xml_path, "xml_model": xml.get("model"),
                    "xml_prompt": xml.get("xml_prompt", ""), "xml_prompt_sha256": xml.get("xml_prompt_sha256", ""),
                    "vision_file": vision_path, "vision_model": vision.get("model"),
                    "vision_prompt_style": vision.get("prompt_style", ""), "source": a.source, **res})

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
