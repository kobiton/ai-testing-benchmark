# AI Testing Benchmark

**Can a vision model find a UI element from a plain-English description?**

If it can, UI tests may not need brittle selectors like:

```java
findElement(By.xpath("//android.widget.Button[@resource-id='btn_scan_qr']"))
```

Instead, a test could describe what it wants:

```java
findElement(byDescription("the scan QR code button"))   // illustrative
```

and let a vision model locate the element on screen.

This repository benchmarks exactly that.

It contains **841 real Android screenshots and 10,307 ground-truth UI elements**, each described in three different ways. 
The runner works with any **OpenAI-compatible endpoint**, including vLLM, llama.cpp, Ollama, and hosted APIs.

It exists because we needed to choose a self-hosted model and wanted a benchmark tailored to the question we actually needed to answer. 
Published so you can check our numbers and score your own model on the same corpus.

---

## Results

**Scored against the human-adjudicated ground truth** (`data/dataset-v1-human.jsonl`): an independent human annotator 
settled every element the two frontier labellings disagreed on and verified the ones they agreed on 
(how it was built is under [The dataset](#the-dataset)). 
Every row is over the same **10,568 elements** — the ones GPT-5.6 described, which every model here has an answer for.

|                            | tap-point correct (centroid) | box correct (IoU ≥ 0.5) | median IoU  | returned a box | asked for |
|----------------------------|------------------------------|-------------------------|-------------|---------------|----------|
| **GPT-5.6**                | **96.4%**                    | **89.7%**               | 0.963       | 100%          | pixels   |
| **Claude Opus 4.8**        | 91.3%                        | 61.2%                   | 0.617       | 100%          | pixels   |
| **Qwen2.5-VL-7B-Instruct** | 84.6%                        | 55.4%                   | 0.553       | 99.98%        | pixels   |
| GUI-Owl-1.5-8B-Instruct    | ⟨TBD⟩                        | ⟨TBD⟩                   | ⟨TBD⟩       | ⟨TBD⟩         | floats   |

The GPT-5.6 row scores the *label* boxes GPT drew as this corpus's co-author — and the human ground truth was made by 
adjudicating between those very boxes — so read it as label quality, not a benchmark answer. 
The other three rows are benchmark answers: screenshot + one description in, box out. 
All three phrasings were asked and `name` is shown; averaged over the three, 
Opus scores 91.97% centroid with 87.07% of elements passing under all three, 
Qwen 85.99% and 75.58% (see [How much the wording matters](#how-much-the-wording-matters)). 
In a result file this table is `summary.by_source.gpt`; a run over the whole file also reports the all-elements figure 
(Qwen: 83.5% centroid over 11,914).

*Deployment:* the Qwen row was measured against a production vLLM serving on a GPU cluster; ⟨TBD GUI-Owl deployment⟩.

**Mind the last column.** GPT-5.6 and Opus were asked for integer pixels, Qwen and GUI-Owl for floats in [0, 1], 
and that is not a free choice — see [How you ask for the coordinates changes the score](#how-you-ask-for-the-coordinates-changes-the-score), 
where the same Opus model scores 7.79% or 90.33% on one 200-element set depending on nothing but that. 
Qwen ignores the instruction and answers in pixels regardless, so its figure is unaffected; 
GUI-Owl answers on a 0–1000 grid and has never been measured under the pixel prompt.

What the human adjudication itself found, in one pass over every kind of disagreement:

- **Hallucination is negligible.** Of the ~3,000 elements only one model found, ~99% are real — the annotator marked *not found* 
  on 0.9% of GPT-only and 0.7% of Opus-only elements. One labeller missing an element is common; inventing one is rare.
- **Where the two labellings agreed** (6,691 elements, box overlap ≥ 0.5 IoU), the human confirmed the agreed position ~98% of the time, 
  and only 9 agreed-on elements (0.13%) turned out not to exist. 
  "Two independent models agree" holds up well as ground truth — with a ~2% blind spot that is now a measurement rather than an assumption.
- **Where they placed the box in genuinely different places** (318 elements), the human sided with GPT about 2:1 — 
  72% of GPT's boxes vs 32% of Opus's reach IoU ≥ 0.5 against the human's.
- The 55 *not found* and 37 *ambiguous* elements are listed, with links to the annotation frames, in 
  [`reference-results/flagged-by-annotator.md`](reference-results/flagged-by-annotator.md).
- Caveat: matched-pair frames carried GPT's phrasing (see the dataset note), which favours GPT's granularity choice on nested pairs. 
  The conflict and one-model-only numbers do not depend on that choice.

**[Browse every row screenshot by screenshot →](https://dataset-review-ui-test.kobiton.com/compare)**
the ground-truth boxes and the model's side by side over each screenshot, every prediction tagged with its IoU and colour-coded by band, and a per-element list with the IoU and centroid verdicts.

### When it is wrong, what did it do?

A failed answer is not one thing. Judged from the centre of the returned box against every human-drawn element on the same screen, 
over the same 10,568 elements and the `name` phrasing as the headline table:

|                                                                  | GPT-5.6 | Claude Opus 4.8 | Qwen2.5-VL-7B | GUI-Owl-1.5-8B |
|------------------------------------------------------------------|---------|-----------------|---------------|----------------|
| right element (centre inside it)                                 | 96.4%   | 91.3%           | 84.6%         | ⟨TBD⟩          |
| **wrong element** — centre inside a *different* labelled element | 1.2%    | 3.2%            | 8.5%          | ⟨TBD⟩          |
| near miss — overlaps the right element, centre just outside      | 2.2%    | 4.6%            | 4.6%          | ⟨TBD⟩          |
| empty space — centre on no labelled element                      | 0.2%    | 1.0%            | 2.3%          | ⟨TBD⟩          |
| declined — answered in prose, no box                             | 0       | 0               | 2 answers     | ⟨TBD⟩          |

GPT-5.6 classifies its *label* boxes against the human's, as in the headline table — boxes drawn while describing the element, 
not answers to a question — so its column is not on the same footing as the other three and reads high. 

When Qwen is wrong it has mostly chosen another control; Opus is wrong less often and, when it is, as likely to have drawn a loose box around the right one. 

This table is about elements that are on the screen, which is the locator's job: given a name, find it. The benchmark asks for a 
box and nothing else, so *declined* counts a model breaking format rather than a considered "not there" — with this prompt, a model that 
cannot find the element still has to guess, and the guess lands in one of the rows above. The complementary question, whether a model 
says "not found" when the element is genuinely absent, is a separate measurement with its own set of absent-element queries; the 
annotator's [92 flagged elements](reference-results/flagged-by-annotator.md) are its starting point. 

Every result file carries the class per row (`answer_class`, with the id of the element chosen instead) and the counts (`summary.answer_classes`), and the 
[viewer](https://dataset-review-ui-test.kobiton.com/compare) filters on them — pick *Wrong element* and each screenshot shows the 
asked-for box and the chosen one side by side. The worst cases of each kind, with those links, are listed per model under 
[`reference-results/failures/`](reference-results/failures/) — what was asked, what was chosen instead — generated by `benchmark/list_failures.py`.

### How much the wording matters

Every element carries three descriptions — `name`, `label`, `intent` — and every model was asked all three. 
The headline table shows `name`; this section is what the other two add. 
The Opus and Qwen figures are on the human ground truth, over the same 10,568 elements as the headline table (Qwen on the production endpoint). 
The GPT-5.6 and GUI-Owl figures still come from the earlier cross-labelled runs — GPT-5.6 scored against the Opus labels (`data/dataset-v1.jsonl`), GUI-Owl as a Q4_K_M GGUF on an Apple Silicon Mac mini
against the same labels — and will move to the human ground truth as those runs are repeated on it. 
The finding does not depend on which answer key is used: it is about how much a model's score moves when only the wording changes.

**Note:** GUI-Owl's centroid sitting above Qwen's is not a better score — it is an answer to a different question. 
It is an agentic model trained to emit a click point `(x, y)`, so 84.9% of its answers carry no box at all. 
A point has no area, so it scores zero on IoU by construction. If what you need is a bounding box, coverage is a **gate**, not a metric, and those two are not on the same leaderboard:
```
iou_accuracy = bbox_coverage × iou_accuracy_given_bbox
     2.93%    =     15.1%     ×        19.4%
```

Asked the same element three ways, every model moves more than the gap between models:

| Phrasing                   | example                                                    | GPT-5.6    | Opus 4.8  | Qwen2.5-VL | GUI-Owl    |
|----------------------------|------------------------------------------------------------|------------|-----------|------------|------------|
| `name` — short common name | *the scan QR code button*                                  | 95.72%     | 91.26%    | 84.60%     | 85.79%     |
| `label` — structural       | *the blue button with the text scan QR code*               | **96.13%** | **93.37%** | **90.17%** | **90.34%** |
| `intent` — functional      | *the button that lets the user scan their sign-in QR code* | 95.00%     | 91.29%    | 83.19%     | 87.28%     |

All four find `label` easiest — it is the phrasing that repeats the element's visible text most often. 
Three of the four then find `intent` hardest; GUI-Owl is the exception, and finds `name` hardest.

A single-phrasing headline flatters a model, because `name` is usually the element's visible text and `intent` deliberately 
carries no text handle at all. 
Scoring all three is what makes the figure describe the ways a tester might actually phrase a locator rather than the easiest one.

**And the average still flatters it.** 
Averaging the three counts an element as 2-of-3 correct; 
what a test suite needs is the element working *whatever* the tester typed. 
Measured per element:

|                          | average centroid | passes under **all three** |
|--------------------------|-----------------|---------------------------|
| GPT-5.6                  | 95.62%          | **92.48%**                |
| Claude Opus 4.8          | 91.97%          | **87.07%**                |
| Qwen2.5-VL-7B-Instruct   | 85.99%          | **75.58%**                |
| GUI-Owl-1.5-8B-Instruct  | 87.80%          | **77.62%**                |

So the figure to plan reliability against is three points below the headline for the strongest model here and
about ten for the open-weight ones — the gap widens as the model weakens, which is the opposite of what an average suggests. 
(`summary.agreement` in every multi-phrasing result file carries these counts.)

Full summaries for every run — token totals and every field described below — are in [`reference-results/`](reference-results/). 
Latency is recorded per run, but each figure describes its own serving setup — it is not comparable across rows, and not a production figure for any of them.

---

## Quickstart

Python 3.9+, and an OpenAI-compatible endpoint serving a vision model.

```bash
git clone https://github.com/kobiton/ai-testing-benchmark.git
cd ai-testing-benchmark

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Validate the dataset and write a result file — calls no model, costs nothing
python benchmark/run_vision_benchmark.py --dry-run
```

Then point it at your model. Start with `--limit` — the full dataset is 10,307 elements, and three phrasings makes that 30,921 requests:

```bash
python benchmark/run_vision_benchmark.py \
    --base-url http://localhost:8000 \
    --model qwen2.5-vl \
    --limit 200
```

`--limit` takes an even stride across the dataset rather than the first N rows, so 200 elements land on ~199 distinct screenshots instead of the first 19. 
It is deterministic, so two pilots are comparable, and a later full run **reuses** the pilot's scores instead of paying for them again.

When the pilot looks sane, drop `--limit` and add the other two phrasings:

```bash
python benchmark/run_vision_benchmark.py \
    --base-url http://localhost:8000 \
    --model qwen2.5-vl \
    --description-index 0 1 2
```

Output lands in `benchmark-results/vision-<model>-GT-<dataset>-<timestamp>.json`. 
Anything that makes a number mean something narrower — a `--limit` pilot, a stopped run, an endpoint that answered as a different model — 
is marked in the filename too; [`benchmark/README.md`](benchmark/README.md) lists them.

---

## Endpoint recipes

Any server exposing `POST /v1/chat/completions` with image content works out of the box. 
`--api-key` is then sent as **both** `Authorization: Bearer` and `X-API-Key` — 
servers disagree about which they read — and omitted entirely when empty.

**vLLM**

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 32768
python benchmark/run_vision_benchmark.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-VL-7B-Instruct
```

**Ollama**

```bash
ollama run qwen2.5-vl
python benchmark/run_vision_benchmark.py --base-url http://localhost:11434 --model qwen2.5-vl
```

**llama.cpp**

```bash
llama-server -m Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj mmproj.gguf --ctx-size 16384
python benchmark/run_vision_benchmark.py --base-url http://localhost:8080 --model qwen2.5-vl
```

**A hosted API** — a strong baseline to compare a self-hosted model against. These need `--api-flavor`, which is **never inferred from the URL**: 
the same base URL can front either shape, and guessing wrong fails per element in a way that reads like a model problem.

```bash
# OpenAI — Bearer auth, and `max_completion_tokens` in place of `max_tokens`
python benchmark/run_vision_benchmark.py --api-flavor openai \
    --base-url https://api.openai.com --api-key "$OPENAI_API_KEY" --model gpt-4o --limit 200

# Anthropic — a different path (/v1/messages), payload and response shape
python benchmark/run_vision_benchmark.py --api-flavor anthropic \
    --base-url https://api.anthropic.com --api-key "$ANTHROPIC_API_KEY" \
    --model claude-opus-4-8 --limit 200
```

> **Pilot first.** A full run is 30,921 requests and the screenshot is essentially the whole input
> at ~3,500 tokens each. `--limit 200` is deterministic and evenly spread, so the pilot and the
> full run are comparable.

> **Context size.** A 1080×2400 screenshot is ~3,300 image tokens on a patch-based vision model,
> which overflows a 4,096-token context on its own. `the request exceeds the available context
> size` comes from the server, not from this script — raise `--ctx-size` / `--max-model-len`.

---

## The dataset

One row per element, coordinates normalised to [0, 1]:

```json
{"screenshot_id": "abc123", "element_id": "e1", "name": "Login", "type": "button",
 "bbox": {"x": 0.12, "y": 0.45, "width": 0.20, "height": 0.06},
 "descriptions": ["the login button",
                  "the button with the text login",
                  "the button that will allow the user to login to the system"]}
```

**The same 841 screenshots are labelled three times: twice, independently, by two different models — and once by a human who adjudicated between them.** 
Same schema, same `screenshot_id`s, same images — so any file works as `--dataset`, 
and scoring one model against the *other* model's labels is what the cross-score in the results above is.

|                    | `dataset-v1.jsonl`                         | `dataset-v1-gpt-5.6.jsonl` | `dataset-v1-human.jsonl` |
|--------------------|--------------------------------------------|---------------------------|-------------------------|
| Labelled by        | Claude Opus 4.8                            | GPT-5.6                   | a human annotator       |
| Elements           | 10,314                                     | 10,647                    | 11,914                  |
| With a bounding box | 10,307 (99.93%)                            | 10,647 (100%)             | 11,914 (100%)           |
| Screenshots covered | 841                                        | 839                       | 841                     |
| Descriptions       | 3 per element (`name` / `label` / `intent`) | same                      | same                    |

The two model labellings disagree about *what an element is*, not only about where its box goes: 
Opus found 10,314 and GPT-5.6 10,647 on the same screens, and there is **no shared element id to join on**. 
Treat them as two opinions, not as one dataset in two files.

The human file resolves that: the two labellings were paired element-by-element by box overlap (~12,000 physical elements), 
every frame was put in front of an independent human annotator on CVAT with both model boxes preloaded, 
and the box they settled on — confirmed, adjusted, or redrawn — is the row. 
Its element ids carry a `gpt-`/`opus-` prefix naming which labelling the element came from, 
and 92 records the annotator flagged (55 *element not found*, 37 *description ambiguous*) are excluded. 
One asymmetry to know: where the two models had matched the same element, the frame showed GPT's phrasings, 
so on nested same-element-different-granularity pairs (the icon vs the row around it) the human was adjudicating GPT's description of it.

Screenshots are 841 Android, 1080×2400 throughout, crawled from **public app-store packages**. 
Every element was found, boxed and described by a frontier model, then spot-checked before the dataset was accepted.

**Ground truth from a model is a stated limitation, not a hidden one.** 
It buys ~10,000 elements instead of the few hundred hand-labelling would have produced, and the box convention is at least *consistent*, 
which is what makes the centroid/IoU split legible. 
The human labelling is what turned that limitation from assumed into measured — 
the label-quality numbers and what the adjudication found are the headline of [Results](#results).

Everything is in the repository: all three `.jsonl` files, their stats sidecars, and all 841 PNGs under `data/images/`, named `<screenshot_id>.png`. 
A clone is ~320 MB and is everything the benchmark needs; no separate download step required.

---

## What is being measured

Two metrics, both always computed. `--metric` only picks which one is the headline.

| Metric       | Pass condition                                                    |         |
|--------------|-------------------------------------------------------------------|---------|
| **centroid** | the centre of the predicted box falls inside the ground-truth box | default |
| IoU          | IoU(pred, gt) ≥ 0.5                                               |         |

**Centroid is the default, and the reason is not that it is easier.** 
It is the question an automated tap actually asks: would the tap land on the control? 
IoU asks something stricter: that the model also draw the same kind of box we do. 
Models tend to box the glyph, while our ground truth boxes the full control including its padding, 
so an element that was located correctly can still fail IoU. 
Centroid separates the two questions; IoU is carried as a secondary box-tightness figure, with the bias stated rather than corrected.

**The benchmark prompt deliberately states no box convention at all.** 
Loading the labeller's rules into it would coach models toward the labeller's answer and measure agreement with our annotator 
rather than element localisation.

### Pixels, or a 0–1000 grid?

**Which coordinate scale a model answers on is a property of the model, and it cannot be recovered from a single answer.** 
On a 1080×2400 screenshot `[67, 91]` is a legal pixel pair *and* a legal point on a 0–1000 grid, 
and the two readings are 2.4× apart on the y axis. 
Qwen2.5-VL answers in the screenshot's own pixels; GUI-Owl answers on a 0–1000 grid.
Reading the grid as pixels gave us this, on a run that looked perfectly clean — 
30,921 rows, 4 errors, every answer short and well-formed:

| GUI-Owl, same 30,921 answers | centroid   | IoU ≥ 0.5 |
|------------------------------|------------|----------|
| read as pixels               | **10.56%** | 0.95%    |
| read on the 0–1000 grid      | **87.80%** | 2.93%    |

So: **pilot 200 elements and read `summary.scale_check` before trusting anything.** 
If it flags, find the model's convention from its model card, add it to `COORD_GRIDS` in `benchmark/scoring/coords.py`, 
and re-score the pilot with `--rescore` rather than paying for the run twice. 
`scale_check` flags and never corrects, and `suspect: false` means "no evidence here" rather than "correct" — 
[`benchmark/README.md`](benchmark/README.md) explains why, and how the check's sensitivity depends on aspect ratio.

### How you ask for the coordinates changes the score

The other half of the same problem: the grid above is how a model's answer is *read*, and this is how it was *asked*. 
`--prompt-style pixels`, the default, states the image size and asks for integer pixels. 
`--prompt-style normalized` asks for floats in [0, 1] and states no size.

That sounds cosmetic and is not. 
A model that ignores the instruction is unaffected — Qwen2.5-VL answers in pixels either way and the parser rescales — 
but a model that *obeys* it can be ruined by it. 
Over 200 elements × 3 phrasings, changing nothing else, one frontier model went from **7.79%** centroid to **90.33%**:

| same 200 elements, same ground truth   | centroid  | IoU ≥ 0.5 | median `dy` |
|----------------------------------------|-----------|-----------|------------|
| `normalized` — floats, no image size   | 7.79%     | 2.27%     | +0.0816    |
| `pixels` — integers, image size stated | **90.33%** | **61.67%** | +0.0004    |

Both runs had zero errors and returned a box on 100% of requests. The model was answering every time; 
under one prompt it was answering in a frame it had never been given — every prediction landing ~8% of the screen height too low. 
That is a harness result, not a model result, and it is the single most expensive thing to get wrong here.

So `prompt_style` is recorded in every result file, and two runs either side of it are not comparable. 
**The two open-weight rows above were measured under `normalized`** and are not understated by it, 
because Qwen ignores the instruction — but GUI-Owl has not been measured under `pixels` at all. 
[`benchmark/README.md`](benchmark/README.md) has the 10-element control that splits the cause into its two halves.

### Screenshots go out at full resolution

`--max-image-dim` defaults to `0`, and downscaling is not a benchmark setting: a comparison is meaningful only when 
every model saw the screenshot a consumer would actually send. 
We learned it expensively — an early run shrank every screenshot to a 1080px long edge and reported **70.07%** centroid for Qwen2.5-VL 
where native resolution reports **84.40%** on the same phrasing, same ground truth, same metric. 
A small control is ~34px across natively and ~15px after that resize, i.e. below what the model can resolve.

Treat a non-zero `max_image_dim` in a result file the way you would treat `PARTIAL` — check what the number describes before quoting it. 
Native costs roughly 4–5× the image tokens, and the image is essentially the whole input; `benchmark/README.md` has the measurements.

---

## Reading a result file

```
=== qwen2.5-vl (primary metric: centroid) ===
  Centroid Accuracy (center inside gt): 85.2%  (26,342/30,921 pass) *
  IoU Accuracy  (IoU≥0.5): 49.2%
  Mean IoU:     0.462
  Median IoU:   0.493
  Errors:       0 (timeout: 0)
```

The JSON carries far more: `summary.by_description` has each phrasing scored separately, 
`summary.agreement` has how often the *same* element passes under one / all / some wordings, 
`summary.by_source` has the figures split by which labelling the element came from (`gpt-`/`opus-` ids — the headline table above is `by_source.gpt`), 
`summary.answer_classes` and each row's `answer_class` say what the model did when it was wrong (see [When it is wrong](#when-it-is-wrong-what-did-it-do)), 
and every row keeps the model's own untruncated answer in `raw` plus the `img_w`/`img_h` it was normalised against. 
`benchmark/README.md` documents every field.

**Read `served_model` before quoting a number.** `--model` is a request, not a guarantee: 
a server holding a single weight file answers every request with whatever is loaded, whatever name you ask for. 
Each run therefore records who actually replied, warns when that disagrees with `--model`, and marks the filename `-SERVED-<name>` — 
so you are told rather than having to check, but the field is still the one that says which model the score belongs to.

Three things the runner does that are worth knowing before a long run:

- **It resumes.** Every scored `(element, phrasing)` pair is checkpointed as it lands, so a crash
  or a closed laptop costs only the requests in flight. Re-run the same command to continue.
  Errors are deliberately *not* checkpointed, so a transient timeout is retried rather than frozen
  into the score. The phrasing is part of the row key, so scoring `name` today and `0 1 2`
  tomorrow pays only for `label` and `intent`.
- **Ctrl+C keeps the score.** The first signal cancels the queue and lets in-flight requests land,
  then writes a real result file marked `-PARTIAL-` in the filename, `"stopped_early": true`
  inside, and `completed_fraction` in the summary. A second signal exits immediately. Without
  those marks a 12%-complete run reads exactly like a finished one.
- **`--rescore` fixes a parsing bug without calling a model.** The answers are all still in `raw`,
  so re-reading them repairs the file: 30,921 rows in ~2 seconds against ~10 hours and 85M input
  tokens for a re-run. Cost is not the main argument — the model is not deterministic, so a re-run
  answers afresh and nobody can then tell a harness fix from a model that replied differently.
  Re-scoring the same `raw` proves which one moved.
- **`--rescore-gt` grades old answers on a ground truth that did not exist when they were made.**
  Same path, answer key swapped: each row's `gt_bbox` is replaced by the given dataset's box for the
  same (screenshot, element), joined with `--rescore-gt-prefix` naming the id prefix that dataset
  gives this run's elements (`gpt` when the run answered `data/dataset-v1-gpt-5.6.jsonl` and the
  human file calls those `gpt-eN`). Rows the new ground truth lacks are dropped, never graded against
  the old box, and each row's description is checked against the dataset's text at the same index.

---

## Labelling your own screenshots

The pipeline that produced the dataset is here too. Four steps, one JSONL out:

```bash
pip install -r requirements-pipeline.txt
cp .env.example .env          # set LLM_PROVIDER and the matching API key

python run_pipeline.py --input path/to/screenshots --output-name my-dataset
```

| Step                 | Calls             |                                            |
|----------------------|-------------------|--------------------------------------------|
| 1 · extract elements | 1 per screenshot  | identify every interactive element         |
| 2 · detect boxes     | **1 per element** | dominates the cost                         |
| 3 · write descriptions | 1 per screenshot  | three phrasings each                       |
| 4 · assemble         | 0                 | merge into one JSONL, write a stats sidecar |

Everything it produces stays on your machine, under `data/`:

```
data/my-dataset.jsonl          the labels, in the same shape as the shipped dataset
data/my-dataset-stats.json     counts, plus which provider and model did the labelling
data/checkpoints/…             per-screenshot progress; deleted once the run completes cleanly
```

Those paths are anchored on the repository root, not on the directory you run the command from,
so a resume finds the earlier run's progress wherever you start it. Point the benchmark at the
result with `--dataset data/my-dataset.jsonl` and `--images-dir path/to/screenshots`.

Budget `N + (N × E) + N` calls for N screenshots averaging E elements. At the 12.3 elements per
screenshot we measured that is ~14 calls each, so the 841-screenshot corpus cost ~12,000 calls.
**Use `--limit` to pilot a prompt change over a couple of hundred calls first** — there is no flag
that runs the pipeline without spending, and a small `--limit` is how you keep a trial cheap.
Runs are checkpointed per screenshot and resume on re-run with the same `--output-name`.

Two warnings the flags do not carry:

- **Labelling is not reproducible.** No `temperature` is sent, because current frontier models
  reject any non-default value, so each model manages its own sampling. Compare boxes within one
  output file, never across two runs.
- **Running this will not reproduce `data/dataset-v1.jsonl` box-for-box.** The step-2 prompt
  has been revised since that file was labelled — same box convention, slightly different boxes
  (median IoU 0.88 against the previous revision on a stratified sample). The shipped dataset is
  the artifact; this pipeline is here to label a *new* corpus, not to regenerate that one.

---

## Repository layout

```
benchmark/
  run_vision_benchmark.py   the benchmark; --help lists every flag
  README.md                 every metric, every field, and why each exists
  cli/ engine/ artifacts/   the flags · the run · checkpoint, result JSON, re-scoring
  model/ scoring/ workload/ the system under test · what an answer means · what gets scored
pipeline/                   the four labelling steps
run_pipeline.py             pipeline entrypoint
data/
  dataset-v1.jsonl          ground truth, labelled by Opus   ← committed
  dataset-v1-gpt-5.6.jsonl  the same screens, by GPT-5.6     ← committed
  dataset-v1-human.jsonl    a human adjudicating the two     ← committed
  images/                   841 screenshots, <id>.png        ← committed
reference-results/          our published summaries          ← committed
```

Both tools write only inside the repository, and everything they write is gitignored:

```
data/<name>.jsonl           a dataset you label yourself
data/<name>-stats.json      its stats sidecar
data/checkpoints/           labelling progress, so an interrupted run resumes
benchmark-results/          one JSON per benchmark run
benchmark-results/checkpoints/   scored (element, phrasing) pairs, so a run resumes
```

Nothing is uploaded anywhere and nothing is read from a network location: the corpus is in the clone, 
and the only address the tools contact is the model endpoint you give them.

---

## Citing this

```bibtex
@misc{kobiton_ai_testing_benchmark,
  title  = {AI Testing Benchmark: natural-language UI element localisation on mobile screenshots},
  author = {Kobiton},
  year   = {2026},
  url    = {https://github.com/kobiton/ai-testing-benchmark}
}
```

---

## License

Code is [Apache-2.0](LICENSE).

The annotations in `data/` are released for research and evaluation use. 
The screenshots depict third-party applications published on public app stores; their interfaces remain the property of their respective owners, 
and they are provided for benchmarking vision models rather than for redistribution as artwork.
