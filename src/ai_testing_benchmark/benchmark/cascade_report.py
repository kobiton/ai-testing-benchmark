"""Score the tree-then-screenshot cascade from an `xml-…` and a `vision-…` result file.

    ai-testing-benchmark cascade --config config/cascade/gpt-5-6-terra.json [--json out.json]
    ai-testing-benchmark cascade benchmark-results/xml-gpt-5.6-terra-….json \\
                                 benchmark-results/vision-gpt-5.6-terra-….json [XML2 VISION2 …] \\
                                 [--annotation-origin gpt-5.6 --dataset data/labels/ground-truth.jsonl]
                                 [--source gpt] [--json cascade.json] [--with-vision-alone]

A cascade config names the two run configs (or two result files outright); with run
configs, the newest complete result of each in the results directory is the input, and
the dataset selection is the one those runs made.

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
import logging
import sys
from pathlib import Path

from ai_testing_benchmark import config as run_config
from ai_testing_benchmark.benchmark.artifacts.cascade import ITEMS, XML_SUBLINES, VISION_SUBLINES, join_cascade
from ai_testing_benchmark.benchmark.workload.dataset import keys_with_origin
from ai_testing_benchmark.paths import display_path, resolve

logger = logging.getLogger(__name__)

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


_SKIP_MARKS = ("-PARTIAL-", "-PILOT-", "-ONLY-FROM-", "-RESCORED-", "-SERVED-")


def _newest_result(results_dir: Path, kind: str, model: str) -> Path:
    """The newest complete `<kind>-<model>-…json` in `results_dir`.

    Complete means none of the filename marks that narrow what a number means — a pilot,
    a stopped run, a subset, an endpoint that answered as another model — because a
    cascade built on one of those would carry the narrowing without the mark.
    """
    safe = model.replace("/", "-").replace(":", "-")
    found = [p for p in results_dir.glob(f"{kind}-{safe}-*.json")
             if not any(m in p.name for m in _SKIP_MARKS)]
    if not found:
        raise FileNotFoundError(f"no complete {kind}-{safe}-*.json in {results_dir}")
    return max(found, key=lambda p: p.stat().st_mtime)


def _from_config(path: str, ap: argparse.ArgumentParser) -> tuple:
    """(xml path, vision path, allowed keys or None, with_vision_alone, resolved config)."""
    cfg, info = run_config.load_cascade(path)
    results_dir = Path(cfg.get("results_dir") or resolve("benchmark-results"))
    halves, dataset_paths = {}, []
    for kind, ckey, rkey in (("xml", "xml_config", "xml_result"), ("vision", "vision_config", "vision_result")):
        if rkey in cfg:
            halves[kind] = Path(cfg[rkey])
        else:
            run_cfg, _ = run_config.load(cfg[ckey], require_key=False)
            if run_cfg["input"] != ("xml" if kind == "xml" else "screenshot"):
                ap.error(f"{ckey} ({display_path(cfg[ckey])}) is not an input={'xml' if kind == 'xml' else 'screenshot'} config")
            dataset_paths.append((run_cfg["dataset.path"], run_cfg.get("dataset.annotation_origin")))
            try:
                halves[kind] = _newest_result(Path(run_cfg.get("output_dir") or results_dir), kind, run_cfg["model"])
            except FileNotFoundError as e:
                ap.error(f"{ckey}: {e} — run it first (ai-testing-benchmark run --config {display_path(cfg[ckey])})")
            logger.info("%s half: %s", kind, display_path(halves[kind]))
    origins = cfg.get("annotation_origin")
    if origins is None and dataset_paths:
        origins = dataset_paths[0][1]
    keys = None
    if origins:
        if not dataset_paths:
            ap.error("annotation_origin needs a run config to take the dataset from (xml_config or vision_config)")
        keys = keys_with_origin(dataset_paths[0][0], origins)
    resolved = {"file": info["file"], "file_sha256": info["sha256"], "model": cfg["model"],
                "xml_result": display_path(halves["xml"]), "vision_result": display_path(halves["vision"]),
                "annotation_origin": list(origins or []), "with_vision_alone": bool(cfg.get("with_vision_alone", False))}
    return halves["xml"], halves["vision"], keys, resolved["with_vision_alone"], resolved


def _select(rows: list, keys) -> list:
    return rows if keys is None else [r for r in rows if (r.get("screenshot_id"), str(r.get("element_id"))) in keys]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ai-testing-benchmark cascade", description=__doc__.split("\n\n")[0])
    ap.add_argument("pairs", nargs="*", metavar="FILE", help="XML result, vision result, repeated per pair")
    ap.add_argument("--config", default="", metavar="FILE",
                    help="a cascade config (config/cascade/<model>.json) naming the two halves; replaces the FILE pair")
    ap.add_argument("--dataset", default="", help="with --annotation-origin: the ground truth whose annotation.origin to select on")
    ap.add_argument("--annotation-origin", nargs="+", default=[], metavar="ORIGIN",
                    help="keep only rows of elements with this annotation.origin in --dataset (e.g. gpt-5.6)")
    ap.add_argument("--source", default="", help="keep only element ids with this prefix (e.g. gpt), as summary.by_source does")
    ap.add_argument("--json", default="", help="also write the numbers here, one object per pair")
    ap.add_argument("--with-vision-alone", action="store_true",
                    help="add the screenshot strategy's own figure on the same rows (needs a full vision run)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    keys, resolved = None, None
    if a.config:
        if a.pairs:
            ap.error("--config names the files; do not also pass a FILE pair")
        try:
            xml_path, vision_path, keys, with_alone, resolved = _from_config(a.config, ap)
        except (run_config.ConfigError, FileNotFoundError) as e:
            ap.error(f"--config: {e}")
        a.pairs = [str(xml_path), str(vision_path)]
        a.with_vision_alone = a.with_vision_alone or with_alone
    elif not a.pairs:
        ap.error("give an XML result and a vision result, or --config")
    if len(a.pairs) % 2:
        ap.error("files come in pairs: XML result then vision result")
    if a.annotation_origin:
        if not a.dataset:
            ap.error("--annotation-origin needs --dataset")
        keys = keys_with_origin(str(resolve(a.dataset)), a.annotation_origin)

    out = []
    for xml_path, vision_path in zip(a.pairs[0::2], a.pairs[1::2]):
        xml, vision = _load(xml_path), _load(vision_path)
        if xml.get("input") != "xml":
            ap.error(f"{xml_path} is not an --input xml result")
        if vision.get("input", "screenshot") != "screenshot":
            ap.error(f"{vision_path} is not a vision result")
        if not xml.get("results") or not vision.get("results"):
            ap.error("both files need their per-request rows; a summary-only file (rows omitted) cannot be joined")
        res = join_cascade(_select(xml.get("results") or [], keys), _select(vision.get("results") or [], keys), source=a.source)
        name = f"{xml.get('model')} (XML) → {vision.get('model')} (vision)"
        print(render(name, xml, vision, res, xml_path, vision_path, with_vision_alone=a.with_vision_alone))
        out.append({"xml_file": display_path(xml_path), "xml_model": xml.get("model"),
                    "xml_prompt": xml.get("xml_prompt", ""), "xml_prompt_sha256": xml.get("xml_prompt_sha256", ""),
                    "vision_file": display_path(vision_path), "vision_model": vision.get("model"),
                    "vision_prompt_style": vision.get("prompt_style", ""), "source": a.source,
                    "annotation_origin": list(a.annotation_origin or (resolved or {}).get("annotation_origin", [])),
                    **({"config": resolved} if resolved else {}), **res})

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
