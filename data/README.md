# The dataset

One ground truth, the screenshots it describes, and their page sources.

```
data/
  labels/ground-truth.jsonl   11,914 human-verified elements on 841 screenshots, one row each
  images/<screenshot_id>.png  the 841 screenshots, 1080×2400 Android
  xml/<screenshot_id>.xml     the 841 page-source dumps (UiAutomator), captured with the screenshot
```

## A row

```json
{"screenshot_id": "a32810f5…", "element_id": "gpt-e28", "name": "L key", "type": "button",
 "bbox": {"x": 0.8528, "y": 0.7425, "width": 0.0861, "height": 0.0521},
 "descriptions": ["the L key",
                  "the keyboard key labeled l",
                  "the key that enters the letter l into the search field"],
 "annotation": {"source": "human-cvat", "origin": "gpt-5.6", "flag": "ok"}}
```

| field               | meaning                                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------------------|
| `screenshot_id`     | names `images/<id>.png` and `xml/<id>.xml`                                                               |
| `element_id`        | unique within the screenshot; the prefix repeats `annotation.origin`                                     |
| `name`, `type`      | the labeller's short name and control type                                                               |
| `bbox`              | the human's box, normalised to [0, 1] of the screenshot's width and height                               |
| `descriptions`      | the three phrasings a run asks: `name`, `label`, `intent` (`--description-index 0 1 2`)                  |
| `annotation.source` | `human-cvat`: the box was confirmed, adjusted or redrawn by a human annotator on CVAT                    |
| `annotation.origin` | which model labelling first found the element and wrote its descriptions: `gpt-5.6` or `claude-opus-4-8` |
| `annotation.flag`   | `ok` for every row here; the annotator's `not_found` and `ambiguous` rows are excluded (below)           |

## Counts

|                                                | elements   | screenshots |
|------------------------------------------------|------------|-------------|
| ground truth                                   | 11,914     | 841         |
| of which `annotation.origin = gpt-5.6`         | **10,568** | 839         |
| of which `annotation.origin = claude-opus-4-8` | 1,346      | -           |
| excluded: flagged `not_found`                  | 55         |             |
| excluded: flagged `ambiguous`                  | 37         |             |

**Every published figure is over the 10,568 `gpt-5.6` rows**: the elements every model here has an answer for. 
The selection is a config value — `dataset.annotation_origin` in the files under [`config/`](../config/), 
`--annotation-origin gpt-5.6` on the command line — not a second label file, so this one file stays the single source of truth. 
The 1,346 `claude-opus-4-8` rows are the elements only the Opus labelling found; 
a run over the whole file reports both groups separately as `summary.by_source.gpt` and `.opus` (keyed by the id prefix).

How the two labellings were paired, adjudicated and what the adjudication found is in [`docs/annotations.md`](../docs/annotations.md); 
the 92 excluded rows are listed in [`docs/annotations/flagged-by-annotator.md`](../docs/annotations/flagged-by-annotator.md).

## The page sources

`xml/<screenshot_id>.xml` is the UiAutomator dump captured at the same moment as the screenshot.
The tree track sends a filtered copy to the model and runs the returned XPath on this full dump.
Two things about it shape every tree figure:

- It covers the **application window only**. A tree read on a live device also holds the on-screen keyboard; 
  here a key on the keyboard has no node, so the ~1,700 elements that are keys (5,060 of the 31,704 requests) 
  cannot be answered from the tree by any model.
- It is the tree **as captured**, not as filtered: the XPath runs on the full dump, as a driver would run it on the live tree.

Two screenshots have no `gpt-5.6` element, so a `gpt-5.6` run touches 839 of the 841.

## Statistics

Nothing is precomputed. Count whatever you need from the file:

```bash
python -c "
import json, collections
rows = [json.loads(l) for l in open('data/labels/ground-truth.jsonl')]
print(len(rows), 'elements on', len({r['screenshot_id'] for r in rows}), 'screenshots')
print(collections.Counter(r['annotation']['origin'] for r in rows))
print(collections.Counter(r['type'] for r in rows).most_common(8))
"
```

`ai-testing-benchmark run --dry-run` loads the file the way a run does, applies the config's
selection and prints how many elements and requests a run would make, without calling a model.

## Earlier files

The two model labellings this ground truth was adjudicated from — `dataset-v1.jsonl` (Claude Opus
4.8, 10,314 elements) and `dataset-v1-gpt-5.6.jsonl` (GPT-5.6, 10,647) — and the earlier
`dataset-v1-human.jsonl` / `dataset-v1-human-gpt.jsonl` pair are no longer in the working tree.
They are kept at git tag `v1`:

```bash
git checkout v1 -- data/dataset-v1-gpt-5.6.jsonl data/dataset-v1.jsonl
```

## Labelling your own screenshots

`ai-testing-benchmark pipeline` writes a dataset of this shape from a directory of screenshots;
the output lands here as `data/<name>.jsonl` (gitignored) with a `data/<name>-stats.json` sidecar.
See [the methodology page](../docs/methodology.md#labelling-your-own-screenshots).

## Licence

The annotations are released for research and evaluation use. The screenshots depict third-party
applications published on public app stores; their interfaces remain the property of their
respective owners, and they are provided for benchmarking models rather than for redistribution.
