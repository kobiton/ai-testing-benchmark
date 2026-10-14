# Annotations — how the ground truth was built

The file is [`data/labels/ground-truth.jsonl`](../data/labels/ground-truth.jsonl), described field by
field in [`data/README.md`](../data/README.md). This page is where its rows came from and what
building it taught us about model-written labels.

## Three labellings of the same 841 screenshots

The screenshots are 841 Android screens, 1080×2400 throughout, crawled from public app-store
packages, each captured together with its UiAutomator page source. They were labelled three times:

|                     | Claude Opus 4.8                             | GPT-5.6       | human annotator |
|---------------------|---------------------------------------------|---------------|-----------------|
| Elements            | 10,314                                      | 10,647        | **11,914**      |
| With a bounding box | 10,307 (99.93%)                             | 10,647 (100%) | 11,914 (100%)   |
| Screenshots covered | 841                                         | 839           | 841             |
| Descriptions        | 3 per element (`name` / `label` / `intent`) | same          | same            |

The two model labellings were produced independently by the labelling pipeline
(`ai-testing-benchmark pipeline`): one call per screenshot to list the interactive elements, one
per element to box it, one per screenshot to write the three descriptions. They disagree about
*what an element is*, not only about where its box goes — Opus found 10,314 and GPT-5.6 10,647 on
the same screens, with no shared element id to join on — so they are two opinions, not one dataset
in two files. Both are kept at git tag `v1` (`data/dataset-v1.jsonl`, `data/dataset-v1-gpt-5.6.jsonl`).

## The adjudication

The human file resolves that. The two labellings were paired element by element by box overlap
(~12,000 physical elements); every frame was put in front of an independent human annotator on
CVAT with both model boxes preloaded, and the box they settled on — confirmed, adjusted, or
redrawn — is the row. Where the two models had matched the same element, the frame showed GPT's
phrasings; where only one model had found the element, that model's. The row's `annotation.origin`
records which (`gpt-5.6` or `claude-opus-4-8`), and the `element_id` prefix repeats it.

The annotator could flag instead of drawing. 55 elements were flagged *not found* — the description
names something that is not on the screen — and 37 *ambiguous* — the description could fit more
than one thing, or nothing decisively. Both kinds are excluded from the ground truth, so no
benchmark figure includes them; they are listed, each with its screenshot and page source, in
[`annotations/flagged-by-annotator.md`](annotations/flagged-by-annotator.md).

One asymmetry to know: on nested same-element-different-granularity pairs (the icon vs the row
around it) the human was adjudicating GPT's description of it, which favours GPT's granularity
choice on those pairs. The conflict and one-model-only findings below do not depend on that choice.

## What the adjudication found

One pass over every kind of disagreement between the two model labellings:

- **Hallucination is negligible.** Of the ~3,000 elements only one model found, ~99% are real — the
  annotator marked *not found* on 0.9% of GPT-only and 0.7% of Opus-only elements. One labeller
  missing an element is common; inventing one is rare.
- **Where the two labellings agreed** (6,691 elements, box overlap ≥ 0.5 IoU), the human confirmed
  the agreed position ~98% of the time, and only 9 agreed-on elements (0.13%) turned out not to
  exist. "Two independent models agree" holds up well as ground truth — with a ~2% blind spot that is
  now a measurement rather than an assumption.
- **Where they placed the box in genuinely different places** (318 elements), the human sided with
  GPT about 2:1 — 72% of GPT's boxes vs 32% of Opus's reach IoU ≥ 0.5 against the human's.
- The not-found set is the one place this corpus shows a labeller inventing an element: 46 of the
  55 carry GPT-5.6's descriptions, 9 Opus's; 16 are the same pattern, a guest-count "increase"
  stepper on one app. It is also the natural seed for measuring whether a model will answer *not
  found* when asked about something absent — see [error-analysis.md](error-analysis.md).

## Why the published figures are over 10,568 elements

The 10,568 rows with `annotation.origin = gpt-5.6` are the elements every model has an answer for:
the Opus screenshot run was made against GPT's descriptions before the adjudication existed, and
its answers were re-graded on the human boxes rather than re-bought. The 1,346 `claude-opus-4-8`
rows are the elements only the Opus labelling found. Runs over the whole file report the two
groups separately (`summary.by_source.gpt` / `.opus`), and the configs under
[`config/`](../config/) select the `gpt-5.6` rows so that every figure is over the same population.

## Ground truth from a model is a stated limitation

The adjudication is what turned that limitation from assumed into measured. Model labelling bought
~10,000 elements instead of the few hundred hand-labelling would have produced, and its box
convention is at least *consistent* — the full control including its padding, which is why the
centroid/IoU split in the results is legible: models tend to box the glyph, so an element that was
located correctly can still fail IoU. The human pass then put a number on how often that labelling
was wrong, and the numbers above are the answer.

## Labelling your own screenshots

The pipeline that produced the model labellings is in the package; the methodology page has the
[recipe](methodology.md#labelling-your-own-screenshots). Its output has the same shape as the
ground truth and can be the `--dataset` of a run, without an `annotation.origin` field.
