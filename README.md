# AI Testing Benchmark

**Can a vision model find a UI element from a plain-English description?**

That question decides whether AI can replace brittle selectors in a test script. Instead of a
locator that breaks the moment a developer renames an id:

```java
findElement(By.xpath("//android.widget.Button[@resource-id='btn_scan_qr']"))
```

you describe the element the way a person would, and something has to look at the screen and
answer *where*:

```java
findElement(byDescription("the scan QR code button"))   // illustrative
```

This repository measures how well models do that. It ships **841 real Android screenshots with
10,307 ground-truth elements**, each phrased three different ways, and a benchmark runner that
works against **any OpenAI-compatible endpoint** - your own vLLM, llama.cpp or Ollama server, or
a hosted API.

It exists because we needed to choose a self-hosted model and could not find a benchmark that
answered our actual question. Published so you can check our numbers, and score your own model
on the same corpus.

---

## Results

Two self-hosted models, both **Q4_K_M GGUF served by Ollama on an Apple Silicon laptop**, every
element scored under all three phrasings — 30,921 requests per model.

| | centroid ★ | IoU ≥ 0.5 | median IoU | returned a box | input tok/element |
|---|---|---|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | **85.19%** | 49.19% | 0.493 | **100%** | 3,551 |
| GUI-Owl-1.5-8B-Instruct | **87.80%** | 2.93% | 0.257 | **15.1%** | 2,747 |

**Read the last two columns before the first one.** GUI-Owl's higher centroid is not a better
score — it is an answer to a different question. It is an agentic model trained to emit a click
point `(x, y)`, so 84.9% of its answers carry no box at all. A point has no area, so it scores
zero on IoU by construction. If what you need is a bounding box, coverage is a **gate**, not a
metric, and the two models are not on the same leaderboard:

```
iou_accuracy = bbox_coverage × iou_accuracy_given_bbox
     2.93%    =     15.1%     ×        19.4%
```

Both models were also asked the same element three ways, which turns out to matter more than the
gap between them:

| Phrasing | example | Qwen2.5-VL | GUI-Owl |
|---|---|---|---|
| `name` — short common name | *the scan QR code button* | 84.40% | 85.79% |
| `label` — structural | *the blue button with the text scan QR code* | **88.11%** | **90.34%** |
| `intent` — functional | *the button that lets the user scan their sign-in QR code* | 83.06% | 87.28% |

A single-phrasing headline flatters a model, because `name` is usually the element's visible text
and `intent` deliberately carries no text handle at all. Scoring all three is what makes the figure
describe the ways a tester might actually phrase a locator rather than the easiest one.

Full summaries, including token totals and every field described below, are in
[`reference-results/`](reference-results/). Latency is recorded but describes a laptop, not a GPU
deployment — do not read it as a production figure.

**[Browse the results screenshot by screenshot →](https://dataset-review-ui-test.kobiton.com/)**

A percentage tells you how often a model was right; it does not show you *how* it was wrong. The
viewer draws the ground-truth box and the model's prediction over the same screenshot — green for
pass, red for fail — with the per-element list beside it and a filter for each of the three
phrasings, so a near-miss on a 34px icon is visibly a different failure from a prediction that
landed on the wrong control entirely. No sign-in, nothing to install, read-only.

---

## Quickstart

Python 3.9+, and an OpenAI-compatible endpoint serving a vision model.

```bash
# The clone is ~315 MB: the screenshots are in the repository, so there is nothing to
# download separately and no dataset registration to sign up for.
git clone https://github.com/kobiton/ai-testing-benchmark.git
cd ai-testing-benchmark

pip install -r requirements.txt

# Validate the dataset and write a result file — calls no model, costs nothing
python benchmark/run_vision_benchmark.py --dry-run
```

Then point it at your model. **Start with `--limit`** — the full dataset is 10,307 elements, and
three phrasings makes that 30,921 requests:

```bash
python benchmark/run_vision_benchmark.py \
    --base-url http://localhost:8000 \
    --model qwen2.5-vl \
    --limit 200
```

`--limit` takes an even stride across the dataset rather than the first N rows, so 200 elements
land on ~199 distinct screenshots instead of the first 19. It is deterministic, so two pilots are
comparable, and a later full run **reuses** the pilot's scores instead of paying for them again.

When the pilot looks sane, drop `--limit` and add the other two phrasings:

```bash
python benchmark/run_vision_benchmark.py \
    --base-url http://localhost:8000 \
    --model qwen2.5-vl \
    --description-index 0 1 2
```

Output lands in `benchmark-results/vision-<model>-<timestamp>.json`.

---

## Endpoint recipes

Any server exposing `POST /v1/chat/completions` with image content works. `--api-key` is sent as
**both** `Authorization: Bearer` and `X-API-Key`, and omitted entirely when empty.

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

**A hosted API** — anything OpenAI-shaped, as a strong baseline to compare a self-hosted model
against:

```bash
python benchmark/run_vision_benchmark.py \
    --base-url https://api.openai.com --api-key "$OPENAI_API_KEY" --model gpt-4o --limit 200
```

> **Context size.** A 1080×2400 screenshot is ~3,300 image tokens on a patch-based vision model,
> which overflows a 4,096-token context on its own. `the request exceeds the available context
> size` comes from the server, not from this script — raise `--ctx-size` / `--max-model-len`.

---

## The dataset

`data/dataset-v1.jsonl` — one row per element, coordinates normalised to [0, 1]:

```json
{"screenshot_id": "abc123", "element_id": "e1", "name": "Login", "type": "button",
 "bbox": {"x": 0.12, "y": 0.45, "width": 0.20, "height": 0.06},
 "descriptions": ["the login button",
                  "the button with the text login",
                  "the button that will allow the user to login to the system"]}
```

| | |
|---|---|
| Screenshots | 841 Android, 1080×2400 throughout, from 100 public app-store packages |
| Elements | 10,314 — 12.3 per screenshot |
| With a bounding box | 10,307 (99.93%) — these are what the benchmark scores |
| Descriptions | 3 per element (`name` / `label` / `intent`), on every element |
| Labelled by | Claude Opus 4.8 |

Screenshots were crawled from **public app-store packages**, not from customer or user data. Every
element was found, boxed and described by a frontier model, then spot-checked before the dataset
was accepted.

**Ground truth from a model is a stated limitation, not a hidden one.** It buys 10,307 elements
instead of the few hundred hand-labelling would have produced, and the box convention is at least
*consistent*, which is what makes the centroid/IoU split above legible. It also means the ceiling
here is agreement with a strong labeller, not with a human.

Both halves are in the repository — `data/dataset-v1.jsonl` and all 841 PNGs under
`data/images/`, named `<screenshot_id>.png`. A clone is ~315 MB and is everything the benchmark
needs; there is no separate download step, no account, and no external host that can go away and
strand the annotations.

---

## What is being measured

Two metrics, both always computed. `--metric` only picks which one is the headline and which rows
are collected into `worst_cases`.

| Metric | Pass condition | |
|---|---|---|
| **centroid** | the centre of the predicted box falls inside the ground-truth box | default |
| IoU | IoU(pred, gt) ≥ 0.5 | |

**Centroid is the default, and the reason is not that it is easier.** It is the question an
automated tap actually asks — would the tap land on the control? IoU asks something stricter, and
on this corpus most of what it measures turns out to be *box convention* rather than localisation.

Take the Qwen2.5-VL run above. **36.0% of all rows locate the element correctly and still fail
IoU.** The mis-sizing is directional rather than noise: predicted area over ground-truth area has a
median of **0.709**, with 46.3% of boxes more than 1.5× too small against 7.0% more than 1.5× too
big. Models box the glyph; our ground truth boxes the full control including its padding.

You can measure how much of the gap that convention accounts for without calling a model again.
Rescale every predicted box around its own centre until its area exactly matches ground truth —
the best case any prompt instruction could reach — and re-score:

| Qwen2.5-VL, same 30,921 answers | IoU ≥ 0.5 | median IoU |
|---|---|---|
| as-is | 49.19% | 0.493 |
| size-corrected | **71.97%** | **0.641** |

So the box convention, not localisation, is the bulk of the IoU shortfall — and a consumer that
tolerated a glyph-tight box would see a very different number from the one in our results table.
That is exactly why the headline is centroid: it separates *did the model find the element* from
*does the model draw the same kind of box we do*. IoU is carried as a secondary box-tightness
figure, with the bias stated rather than corrected.

**The benchmark prompt deliberately states no box convention at all.** Loading the labeller's
rules into it would coach models toward the labeller's answer and measure agreement with our
annotator rather than element localisation.

### Five fields to read before believing an accuracy

Each of these exists because a number was once believed that should not have been.
[`benchmark/README.md`](benchmark/README.md) has the full story on every one.

| Field | Why it decides whether the accuracy means anything |
|---|---|
| `bbox_coverage` | Share of answers that contained a box at all. A **gate** where you need a box — see the results table |
| `scale_check` | Whether predictions are distributed like the ground truth per axis. The only guard that can see a coordinate-convention error, and only in aggregate |
| `clamped_pred_count` | Predictions pushed against the [0,1] edge. Catches *prose scraped into a box* — 23% on one thinking model we tried, 1.0% on the Qwen run. A **low** value means nothing: a full-width list row genuinely has width 1.0 |
| `workers` | The concurrency every latency figure was measured at. Two runs at different values are not comparable on latency |
| `elements_with_token_counts` | Kept apart from the token totals, because a server reporting no `usage` block leaves them at 0 — not the same finding as a cheap model |

### If you are scoring your own model, read this first

**Which coordinate scale a model answers on is a property of the model, and it cannot be recovered
from a single answer.** On a 1080×2400 screenshot `[67, 91]` is a legal pixel pair *and* a legal
point on a 0–1000 grid, and the two readings are 2.4× apart on the y axis. Qwen2.5-VL answers in
the screenshot's own pixels; GUI-Owl answers on a 0–1000 grid. Reading the grid as pixels gave us
this, on a run that looked perfectly clean — 30,921 rows, 4 errors, every answer short and
well-formed:

| GUI-Owl, same 30,921 answers | centroid | IoU ≥ 0.5 |
|---|---|---|
| read as pixels | **10.56%** | 0.95% |
| read on the 0–1000 grid | **87.80%** | 2.93% |

So: **pilot 200 elements and read `summary.scale_check` before trusting anything.** If it flags,
find the model's convention from its model card, add it to `COORD_GRIDS` in
`benchmark/run_vision_benchmark.py`, and re-score the pilot with `--rescore` rather than paying for
the run twice. `scale_check` flags and never corrects, and `suspect: false` means "no evidence
here" rather than "correct" — [`benchmark/README.md`](benchmark/README.md) explains why, and how
the check's sensitivity depends on aspect ratio.

### Screenshots go out at full resolution

`--max-image-dim` defaults to `0`, and downscaling is not a benchmark setting: a comparison is
meaningful only when every model saw the screenshot a consumer would actually send. We learned it
expensively — an early run shrank every screenshot to a 1080px long edge and reported **70.07%**
centroid for Qwen2.5-VL where native resolution reports **84.40%** on the same phrasing, same
ground truth, same metric. A small control is ~34px across natively and ~15px after that resize,
i.e. below what the model can resolve.

Treat a non-zero `max_image_dim` in a result file the way you would treat `PARTIAL` — check what
the number describes before quoting it. Native costs roughly 4–5× the image tokens, and the image
is essentially the whole input; `benchmark/README.md` has the measurements.

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
`summary.agreement` has how often the *same* element passes under one / all / some wordings, and
every row keeps the model's own untruncated answer in `raw` plus the `img_w`/`img_h` it was
normalised against. `benchmark/README.md` documents every field.

**Read `served_model` before quoting a number.** `--model` is a request, not a guarantee: a server
holding a single weight file answers every request with whatever is loaded, whatever name you ask
for. Each run therefore records who actually replied, warns when that disagrees with `--model`, and
marks the filename `-SERVED-<name>` — so you are told rather than having to check, but the field is
still the one that says which model the score belongs to.

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

---

## Labelling your own screenshots

The pipeline that produced the dataset is here too. Four steps, one JSONL out:

```bash
pip install -r requirements-pipeline.txt
cp .env.example .env          # set LLM_PROVIDER and the matching API key

python run_pipeline.py --input path/to/screenshots --output-name my-dataset
```

| Step | Calls | |
|---|---|---|
| 1 · extract elements | 1 per screenshot | identify every interactive element |
| 2 · detect boxes | **1 per element** | dominates the cost |
| 3 · write descriptions | 1 per screenshot | three phrasings each |
| 4 · assemble | 0 | merge, stats, optional S3 upload |

Budget `N + (N × E) + N` calls for N screenshots averaging E elements. At the 12.3 elements per
screenshot we measured that is ~14 calls each, so the 841-screenshot corpus cost ~12,000 calls.
**Use `--limit` to pilot a prompt change over a couple of hundred calls first.** Runs are
checkpointed per screenshot and resume on re-run with the same `--output-name`.

Three warnings the flags do not carry:

- **`--dry-run` still makes billable calls.** It only skips the S3 upload. The *benchmark's*
  `--dry-run` is the one that calls nothing.
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
pipeline/                   the four labelling steps
run_pipeline.py             pipeline entrypoint
data/
  dataset-v1.jsonl          the ground-truth labels
  images/                   841 screenshots, <screenshot_id>.png
reference-results/          our published summaries, for checking your numbers
```

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

The annotations in `data/` are released for research and evaluation use. The screenshots depict
third-party applications published on public app stores; their interfaces remain the property of
their respective owners, and they are provided for benchmarking vision models rather than for
redistribution as artwork.
