# Benchmark reference

Companion to the [root README](../README.md), which covers installing, getting the data and running.
This file is the reference: every metric, every field in the output, and the reasoning behind the
parts that are not obvious.

Most sections here exist because a number was once believed that should not have been. They are
written as warnings rather than as description for that reason.

---

## Where the code is

`run_vision_benchmark.py` is a ~45-line entry point over a package, and **the layers depend
downwards only**:

```
benchmark/
├── run_vision_benchmark.py   entry point — puts the repo root on sys.path, calls cli.commands
├── cli/         args.py (every flag) · commands.py (dispatch)
├── engine/      runner.py — thread pool, per-element step, stop handling
├── artifacts/   checkpoint.py · builder.py (build_result) · rescoring.py
├── model/       prompts.py · images.py · client.py — the system under test
├── scoring/     coords.py · parsing.py · metrics.py — what an answer means
└── workload/    dataset.py · phrasings.py · sampling.py — what gets scored
```

`workload` and `scoring` depend on nothing internal; `model` on `scoring`; `artifacts` on
`scoring`, `model`, `workload`; `engine` on all of those; `cli` on the rest. **An import pointing
the other way is what would make this circular**, so check the direction before adding one.

There is no test suite. Changes are checked against behaviour instead, and `--rescore` is by far
the cheapest and strongest check: it re-reads a finished result's own answers, calls no model, and
is deterministic, so it drives the parser, the metrics and `build_result` over every row of a real
run in about two seconds. Compare two runs by keying rows on
`(screenshot_id, element_id, description_index)` — `results` is in thread-completion order, so
comparing by position reports dozens of differences on an identical build.

---

## Metrics

Both are always computed. `--metric` only selects the headline number.

| Metric | Pass condition | |
|--------|----------------|---|
| **centroid** | Centre of the predicted bbox falls inside the ground-truth bbox | default |
| iou | IoU(pred, gt) ≥ 0.5 | |

Centroid is the default because it is the question a tap asks, and because IoU on this corpus is
dominated by box-convention disagreement rather than by localisation — the root README has the
figures.

**But centroid cuts both ways, so a third figure is needed to keep two models comparable.** A model
that never returns a box cannot be judged on IoU, and a model judged only on centroid is never asked
for the box the consumer wanted. `summary.bbox_coverage` is that figure — the share of requests that
came back with a bounding box at all — and it factors the IoU exactly:

```
iou_accuracy = bbox_coverage x iou_accuracy_given_bbox
```

| | centroid | bbox_coverage | IoU given a box | = IoU |
|---|---|---|---|---|
| GUI-Owl 1.5 8B | **87.8%** | **15.1%** | 19.4% | 2.9% |
| Qwen2.5-VL | 85.2% | **100%** | 49.2% | 49.2% |

Read left to right that is one sentence: GUI-Owl finds the element slightly more often, and supplies
the box in one request out of seven. **Where the consumer requires a bounding box, coverage is a gate
rather than a metric** — a model below it is not comparable on centroid with one above it, and the
87.8% is not a better score than the 85.2%, it is an answer to a different question. Both figures
were derivable from `point_only_count` before they were named, which is precisely why neither
appeared in any earlier report.

---

## Which model actually answered

**`--model` is a request, not a guarantee.** A server holding one weight file does not route by name:
ask it for `gemma-4-12b` while Qwen is loaded and it returns a perfectly normal Qwen answer with no
error. That is how two of our runs produced result files with **byte-identical predictions on all
112 elements** — one model scored twice under two names. llama.cpp's `llama-server` and any thin
proxy in front of a single backend behave this way; vLLM, Ollama and hosted APIs do route by name.

Nothing is left for you to remember. Before scoring anything, every run:

- lists what the endpoint admits to holding (`_list_loaded_models`, `GET /v1/models`);
- sends one 16-token ping and reads the `model` field back out of the reply
  (`_probe_served_model`) — the response's own name is the only thing that tells the truth.
  16 rather than 1 because a reasoning model spends the cap before it emits anything, and OpenAI
  answers a cap it cannot finish under with a **400** rather than a truncated reply: measured,
  `max_completion_tokens: 1` is a 400 on a current GPT model and 16 is a 200;
- records it as `served_model`, logs a loud warning when it disagrees with `--model`, and puts
  `-SERVED-<model>` in the output filename, so a mismatched run cannot sit in a results directory
  looking like the model you asked for;
- refuses outright when several `--model` names are given and the endpoint has exactly one loaded,
  since that would produce N identically-scored files and a comparison table of one model against
  itself.

The served name is also what picks the coordinate grid (see below), which is why the probe happens
before the run rather than being an optional pre-flight: on a single-model endpoint the served id,
not the requested one, is the fact that decides how its answers must be read.

---

## All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/dataset-v1.jsonl` | Path to the labelled dataset |
| `--images-dir` | `data/images` | Directory of screenshots |
| `--base-url` | `$BASE_URL` or `http://localhost:8080` | Base URL of an OpenAI-compatible server. `--proxy-url` is an alias |
| `--api-key` | `$API_KEY`, else empty | On the default flavor, sent as both `Authorization: Bearer` and `X-API-Key`; omitted entirely when empty |
| `--api-flavor` | `proxy` | Which dialect the endpoint speaks: `proxy`, `openai` or `anthropic`. **Never inferred from the URL** — see below |
| `--prompt-style` | `pixels` | How coordinates are asked for: `pixels` (states the image size, asks for integers) or `normalized` (floats in [0,1]). Changes the score more than anything else here — see below |
| `--model` | `qwen2.5-vl` | Model name(s) — space-separated for several |
| `--metric` | `centroid` | Primary metric: `centroid` or `iou` |
| `--workers` | `3` | Concurrent requests |
| `--description-index` | `0` | Which phrasing(s) to ask: `0`=name, `1`=label, `2`=intent. Takes several — `0 1 2` scores every element under all three |
| `--timeout` | `120` | Per-request timeout in seconds; thinking models need 120s+ |
| `--coord-format` | `corner` | How to read `(x, y)`: `corner` = top-left, `center` = element centre |
| `--thinking-budget` | `-1` | `budget_tokens` for CoT models; `0` disables thinking, `-1` leaves the model default. Sent on `--api-flavor proxy` only |
| `--max-image-dim` | `0` | `0` sends the original. **Not a benchmark setting** — see below |
| `--max-tokens` | `2048` | Output cap. A truncated answer is recorded as an error, not a bbox |
| `--temperature` | `0` | Greedy, so a coordinate is reproducible. `-1` sends none |
| `--limit` | `0` | Score at most N **elements**, spread evenly across the dataset. Deterministic |
| `--coord-grid` | `-1` | Divide pixel-scale answers by N on both axes. `-1` picks it per model from `COORD_GRIDS`; `0` forces the screenshot's own pixels |
| `--no-resume` | off | Discard the checkpoint and score everything again |
| `--finalize-only` | off | Write a `PARTIAL` result from the existing checkpoint and exit — calls no model |
| `--rescore` | — | Re-score a finished result file from its own `raw` text and exit. Calls no model |
| `--output-dir` | `benchmark-results/` | Where result JSON files go |
| `--dry-run` | off | Validate the dataset without calling the API |

Path defaults are anchored on the repository root, so `python benchmark/run_vision_benchmark.py`
works from anywhere in the tree.

### What the output filename tells you

Everything that makes a number mean something other than "this model, this dataset, all of it" is
in the name, because a results directory is read as a list of filenames:

```
vision-<model>[-Nphrasing][-SERVED-<x>][-PARTIAL][-PILOT-<n>]-GT-<dataset>-<timestamp>.json
```

| infix | when |
|---|---|
| `-Nphrasing` | more than one `--description-index`; the headline averages them |
| `-SERVED-<x>` | the endpoint answered as a different model than `--model` asked for |
| `-PARTIAL` | the run was stopped or rebuilt with `--finalize-only` |
| `-PILOT-<n>` | `--limit` cut the run short of the dataset; `n` is how many elements it covers |
| `-RESCORED` | written by `--rescore`, from an existing file's own answers |
| `-GT-<dataset>` | **always** — which ground truth the score is against |

`-GT-` is unconditional because there is no single obvious ground truth to stay quiet about:
`--dataset` points at the shipped corpus by default but at whatever you labelled yourself the
moment you use the pipeline. Every infix goes *before* the timestamp, so the run time stays the
last thing in the name and a listing sorted by it stays sorted by when the model was called.

---

## `--api-flavor` — three dialects, chosen explicitly

| flavor | endpoint | path | auth | output cap |
|---|---|---|---|---|
| `proxy` (default) | vLLM, llama.cpp, Ollama, gateways | `/v1/chat/completions` | `Bearer` **and** `X-API-Key` | `max_tokens` |
| `openai` | api.openai.com | `/v1/chat/completions` | `Bearer` | `max_completion_tokens` |
| `anthropic` | api.anthropic.com | `/v1/messages` | `x-api-key` + `anthropic-version` | `max_tokens` |

**It is never inferred from the URL.** The same base URL can front either shape, and a wrong guess
fails per element in a way that reads like a model problem rather than a config one. The default
sends both auth headers precisely so that no flag is needed for the self-hosted case: a server that
does not recognise one simply ignores it, and an empty `--api-key` sends neither.

**Two of these were found only by calling the real API, and a stub cannot find them** — a stub
encodes the same assumption the code does, so it passes and proves nothing:

- **OpenAI renamed the output cap.** Its current models answer `max_tokens` with *"Unsupported
  parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."*
  vLLM and llama.cpp know only the old name and Anthropic requires it, so the key is split per
  flavor rather than renamed globally.
- **The served-model probe needs headroom.** It used to ask for 1 token, which is fine on llama.cpp
  and a **400** on a reasoning model, because OpenAI refuses a cap it cannot finish under rather
  than truncating.

Anthropic differs in everything, and three details are load-bearing:

- The system prompt is top-level `system`, not a message, and the image is
  `{"type": "image", "source": {...}}` rather than an `image_url`.
- **`budget_tokens` is not sent**, on `anthropic` *or* `openai`. Anthropic spells thinking
  `{"thinking": {...}}` and OpenAI rejects an unrecognised argument outright, so leaving it in
  would turn `--thinking-budget` into a 400 on every element. It goes to `proxy` only, which is the
  only place it ever did anything.
- `stop_reason: "max_tokens"` is mapped onto `finish_reason: "length"` so the truncation guard
  covers it — the guard being what stops a thinking model's prose reaching the parser and being
  scraped into a bounding box.
- Anthropic reports cache reads *outside* `input_tokens`, so they are added back. Without that a
  cached element looks cheaper than it was, and the token totals are what a cost comparison is read
  off.

**A full run against a hosted model is real money.** 10,307 elements at three phrasings is 30,921
requests, and the image is essentially the whole input at ~3,500 tokens each. Pilot with
`--limit 200` first; it is deterministic and evenly spread, so the pilot and the full run are
comparable.

---

## `--prompt-style` — the coordinate format is a model-selection variable, not a detail

Two wordings of the same request, both in `model/prompts.py`:

| style | asks for | states image size |
|---|---|---|
| `pixels` (default) | integers in the screenshot's own pixels | yes |
| `normalized` | floats in `[0,1]` | no |

The harness used to ask for floats and state no size. That is a fine request for a model that
ignores it — Qwen2.5-VL answers in pixels regardless, 5 of 6 raws on a pilot, and the parser
rescales — and a ruinous one for a model that obeys.

| 10-element control, one frontier model | centroid | median `dy` | median `dx` | box height vs GT |
|---|---|---|---|---|
| `normalized` — floats, no image size | **0/10** | +0.0988 | +0.0256 | 1.41× |
| `normalized` + the one `IMAGE SIZE` line | 4/10 | +0.0124 | +0.0256 | — |
| `pixels` — integers, image size stated | **10/10** | +0.0025 | +0.0002 | 0.82× |

Read the `dx` column against the `dy` one: stating the size fixes most of the *vertical* error and
does not touch the *horizontal* one at all; asking for integers fixes both.

Ten elements is enough to explain a bad score and to justify changing the default; it is not a
measurement. So the same model was re-run over **200 elements × 3 phrasings = 600 requests**,
against `data/dataset-v1-gpt-5.6.jsonl` — the second labelling, so the model is not scored on its
own output — changing nothing but the coordinate format:

| same 200 elements, same ground truth | centroid | IoU ≥ 0.5 | mean IoU | median `dy` | box height vs GT |
|---|---|---|---|---|---|
| `normalized` | 7.79% | 2.27% | 0.047 | +0.0816 | 1.69× |
| `pixels` | **90.33%** | **61.67%** | **0.547** | +0.0004 | 0.99× |

No errors, and a bounding box on 100% of requests, in **both** runs. The model was answering every
time; under one prompt it was answering in a frame it had never been given. At full scale — 30,921
requests — the positional error stays under 0.001 of the screen on both axes, so this is not an
artefact of the sample size.

Both halves of that are reproducible from a clone:

```bash
python benchmark/run_vision_benchmark.py --api-flavor anthropic \
    --base-url https://api.anthropic.com --api-key "$ANTHROPIC_API_KEY" \
    --model claude-opus-4-5 --dataset data/dataset-v1-gpt-5.6.jsonl \
    --description-index 0 1 2 --limit 200 --prompt-style normalized

# then the same command with --prompt-style pixels; the checkpoint forks by style,
# so the second run pays for itself rather than resuming the first
```

**The model in that table is not the one in the root README's results table**, which is a different
vendor's and was measured under `pixels` throughout. Two frontier models do not respond to this the
same way, which is exactly why the style is a flag and is recorded in every result.

Two facts kept this from being read as *"that model is bad at grounding"*, and both are worth
reproducing before anyone reports on a model from this harness:

- The labelling pipeline (`pipeline/step2_detect_bboxes.py`) asks the **same model** for pixel
  integers and gets labels that agree with an independent labeller's to four decimals. Same model,
  same screenshots, different question.
- The fault splits in two, per the table above, and neither half alone accounts for the result. A
  single-variable story would have been wrong, which is why the fix was to switch prompts rather
  than add a line to the existing one. The control row is not in the code — it is a diagnostic,
  not something to benchmark under — but it is a two-line insertion described above
  `PROMPT_STYLES` if it ever needs reproducing.

Consequences that are easy to get wrong:

- **The two published open-weight numbers are not understated.** Qwen's 85.19% and GUI-Owl's 87.80%
  were measured under `normalized`, but Qwen ignores the instruction, so its number is unaffected.
  GUI-Owl answers on a 0–1000 grid and **has not been measured under `pixels`** — pilot before
  assuming either way. The GPT-5.6 row beside them was measured under `pixels`, which is why the
  results table carries a column saying so.
- **`prompt_style` is recorded in every result, and defaulted nowhere it describes history.** The
  runtime default is `pixels`; the fallbacks that describe an *existing* artifact (`--rescore`, the
  `--finalize-only` header read) stay `normalized`, because a file predating the flag can only have
  come from the one prompt there was. `build_result` takes it as a **required keyword-only**
  argument for the same reason — any default there would stamp a guess into an artifact as fact.
- **Checkpoints fork by style.** Sharing one would let a second run under another style resume from
  the first, skip every element and report the first run's predictions as its own — visible only as
  "N already scored". `normalized` maps to no suffix, so any checkpoint written before the flag
  existed keeps its path.

---

## Interrupting, resuming, recovering

### Resuming

Every scored `(element, phrasing)` pair is appended to
`<output-dir>/checkpoints/<dataset>-<model>[-<prompt-style>].partial.jsonl` as it finishes, so a
kill, a crash or a sleeping laptop costs only the pairs in flight. **To continue, re-run the same
command** — same dataset, model and prompt style:

```
INFO Resuming from .../dataset-v1-qwen2.5-vl-pixels.partial.jsonl — 6218 (element, phrasing) pair(s) already scored
INFO   6218 already scored, 4089 to go
```

Five rules that are not obvious:

- **Errored elements are not checkpointed**, so a resume retries them. Freezing a transient timeout
  into the score as a permanent failure would understate the model. Whatever still errors on the
  final pass is reported as an error. A run that had errors keeps its checkpoint so the next one
  retries just those.
- **Swapping the loaded model and resuming is refused.** The checkpoint header pins the model that
  actually answered; resuming under a different one would blend two models into one score. Re-load
  that model, or pass `--no-resume`.
- **`--limit` is not part of the checkpoint identity.** An even-stride subset is drawn from the same
  rows, so a 2,000-element pilot's scores are reused by a later full run rather than thrown away.
- **Neither is `--description-index`.** The phrasing is part of each row's key instead, so scoring
  `name` today and `0 1 2` tomorrow pays only for `label` and `intent`. It used to be a `-d0` suffix
  on the filename, which made the second run open a different file and re-score all 10,307 `name`
  elements it already had.
- **`--prompt-style` *is*, and so is `--coord-grid`.** Those rows are not reusable — they came from
  a different question, or were normalised under a different convention — and no field on a row says
  which. The style forks the filename; a grid mismatch is refused outright with the fix in the error.

### Stopping a run and keeping its score

Resuming assumes you will finish. Often you will not: the model is clearly bad, or you have seen
enough of a 30,000-request run. Stopping used to leave the checkpoint but **no result file**, and the
result file is the artifact.

**SIGTERM or the first Ctrl+C winds the run down instead of killing it.** The queue is cancelled,
requests already in flight are allowed to land, and a result file is written with `-PARTIAL-` in its
name and `"stopped_early": true` inside. Send the signal a second time to exit immediately — worth
knowing, because "in flight" is up to one `--timeout` per worker.

`summary.selected_elements` and `summary.completed_fraction` record how much of the run actually
happened. **Read them:** every accuracy in the file is over the elements that were scored, not over
the dataset, so a run stopped at 12% looks exactly as confident as one that finished.

A partial run's checkpoint is always kept, so re-running carries on rather than paying twice.

### Recovering a run that was killed outright

A crash, an OOM kill or a Ctrl+C that reaches the whole process group leaves the checkpoint but no
result file and no chance to wind down. Rebuild the result from what is on disk — no model is called
and nothing is charged:

```bash
python benchmark/run_vision_benchmark.py --finalize-only --model qwen2.5-vl
```

Pass the same `--dataset`, `--model`, `--limit` and `--prompt-style` as the run that died; those
identify its checkpoint. `--description-index`, `max_tokens`, `workers`, `coord_grid`,
`prompt_style` and `served_model` are then read back out of the checkpoint *header* rather than the
command line, because the values that shaped those predictions are the dead run's, not whatever the
flags say at recovery time.

---

## Coordinate space — the same numbers can mean two things

A model that grounds an element answers with numbers. **Which scale those numbers are on is a
property of the model, and it is not recoverable from a single answer.** On a 1080×2400 screenshot
`[67, 91]` is a legal pixel pair *and* a legal point on a 0–1000 grid, and the two readings are 2.4×
apart on the y axis.

Qwen2.5-VL answers in the screenshot's pixels. **GUI-Owl answers on a 0–1000 grid** — it is a
Qwen-VL-family model and inherits that convention. Reading its grid values as pixels shrinks x by
1.08× and y by 2.4×, which produced this:

| GUI-Owl, 30,921 answers | centroid | IoU≥0.5 |
|---|---|---|
| as recorded (read as pixels) | **10.56%** | 0.95% |
| re-scored on the 0–1000 grid | **87.80%** | 2.93% |

Nothing in that run looked wrong: 30,921 rows, 4 errors, every answer short and well-formed,
`clamped_pred_count` at 0.6%. The run was clean and the number it reported was a property of the
harness. Three rules follow, and they are the whole design:

- **`COORD_GRIDS` (in `scoring/coords.py`) is an explicit per-model table, matched on the requested
  *and* served name.** Never
  inferred per answer. A single-model endpoint ignores the name it is asked for, so the served id
  decides — the same reason `_probe_served_model` exists.
- **The grid is implemented by handing it to the parser as both dimensions** (`_norm_dims`). There is
  no second code path, so the pixel and grid readings cannot drift apart, and the "already in [0,1]"
  branches keep working — load-bearing, because GUI-Owl mixes normalised floats
  (`[0.19, 0.81, 0.81, 0.85]`) into the same run as grid integers (`[67, 91]`).
- **1000, not 999.** Both fit; the sweep gives 85.62% at /1000 against 85.91% at /999. 1000 is the
  documented Qwen-VL convention and 999 is merely the value that scores highest on this dataset.
  Fitting a constant to the benchmark it is then evaluated on is not worth 0.29pp.

`coord_grid` is recorded in every result and pinned in the checkpoint header. Resuming a checkpoint
under a different grid is **refused**, not silently blended: no field on a row says which convention
produced it, so a mixed file is undetectable afterwards.

### `summary.scale_check` — the aggregate tell

Every guard that looks at answers one at a time is blind here, because a grid value and a pixel value
are individually indistinguishable. The tell exists only in aggregate: over thousands of rows a model
that finds elements at all must scatter its answers roughly the way the targets are scattered, so
`scale_check` compares the median predicted centroid against the median ground-truth centroid per
axis.

```
GUI-Owl as recorded   x_ratio 0.924   y_ratio 0.441   axes_disagree true    suspect TRUE
GUI-Owl re-scored     x_ratio 0.998   y_ratio 0.981   axes_disagree false   suspect false
Qwen2.5-VL            x_ratio 1.009   y_ratio 0.977   axes_disagree false   suspect false
```

Axis *disagreement* is the sharper of the two signals, because one wrong divisor hits the longer side
harder — 1.08× on x against 2.4× on y. A model with an honest bias toward the top of the screen is
wrong in the same direction on both axes. Below `SCALE_CHECK_MIN_ROWS` (100) the numbers are reported
but no verdict is reached.

**It flags and never corrects.** Guessing a convention from a statistic and applying it silently is
how this harness produced its worst data; a wrong guess would be indistinguishable from a right one
downstream.

**Its sensitivity depends on aspect ratio, so a quiet result is not a clean bill of health.** The
signal is the gap between the true divisor and the one used: on a 1080×2400 screenshot a 0–1000 grid
is 2.4× out on y and the check screams, but on a near-square 862-tall screenshot it is 0.86× out —
inside the band, no verdict. Treat `suspect: false` as "no evidence here", and settle a model's
convention from `COORD_GRIDS` or its model card, not from one run's check.

---

## `--rescore` — repair a result without re-running it

A coordinate-convention bug is a *parsing* bug. The model's answers were fine and are all still in
`raw`, so fixing the parser and re-reading them is the entire repair:

```bash
python benchmark/run_vision_benchmark.py \
    --rescore benchmark-results/vision-gui-owl-1.5-8b-20260802-124638.json
# 30,921 rows re-scored in 1.9s — centroid 10.56% -> 87.80%
```

The fourth path into `build_result`, alongside a completed run, a stopped one and `--finalize-only`,
so its output is shape-identical. Re-running instead would cost **~10 hours and 85M input tokens** —
but the real argument is evidence, not cost: the model is not deterministic, so a re-run answers
afresh and nobody can then tell a harness fix from a model that happened to reply differently.
Re-scoring the same `raw` proves which one moved.

- **Carried over untouched:** tokens, latency, `served_model`, `stopped_early`, the phrasings, `date`
  (when the inference happened), and **every non-parse error**. Only the coordinates and the pass/fail
  drawn from them are recomputed.
- **Only parse-family errors are re-parsed** (`_REPARSEABLE_ERROR_PREFIXES`). A timeout, an HTTP 500,
  an `image not found` or a `Truncated at max_tokens=…` describes something that happened *instead of*
  an answer. The truncation case is why the list exists rather than a blanket re-parse: a thinking
  model that spends its budget leaves reasoning prose in `raw`, and handing that to the parser is
  exactly the bug that recorded one run at 16%.
- **Never overwrites the input.** Output carries a `-RESCORED-` infix *before* the timestamp, like
  `-PARTIAL-` and `-SERVED-`, so the run time stays the last timestamp in the name.
- **Dimensions come from the row's own `img_w`/`img_h` first**, then from the image on disk with
  `--max-image-dim` re-applied. With a `--coord-grid` in force they are not consulted at all, so a
  grid rescore needs nothing but the result file.

Proof the path is faithful: re-scoring the Qwen2.5-VL run with the pixel reading it already used
moved **0 of 30,921 predictions** and reproduced every figure exactly (centroid 85.19% → 85.19%, IoU
49.19% → 49.19%). Worth repeating after any parser change.

---

## Phrasings — asking the same element three ways

The labelling pipeline writes **three** descriptions per element:

| # | Style | Example |
|---|---|---|
| 0 | `name` | *the login button* |
| 1 | `label` | *the button with the text login* |
| 2 | `intent` | *the button that will allow the user to login to the system* |

The benchmark used to ask only `descriptions[0]`, so two thirds of a dataset that had already been
paid for went unused — and the third being measured is the easiest one. `name` is usually the
element's **visible text**, so it flatters a model that grounds by reading text; `intent`
deliberately carries no text handle.

```bash
# Score every element under all three, on a 1,500-element pilot = 4,500 requests
python benchmark/run_vision_benchmark.py --model qwen2.5-vl --description-index 0 1 2 --limit 1500
```

Four things to know:

- **`--limit` counts elements, not requests.** N phrasings means N times the calls.
- **`total_elements` becomes rows** — elements × phrasings. `unique_elements` is the element count,
  and every accuracy in `summary` is over the rows, so the headline *averages* the phrasings.
  Per-phrasing figures are in `by_description`; that is where the finding is.
- **`agreement` only counts elements scored under every phrasing.** A stopped run leaves elements
  holding one or two of their three, and folding those in would score a 1-of-1 as unanimous. `any` is
  the ceiling a better prompt could reach — the element was findable, some wording found it. `all` is
  robustness. `mixed` is the population worth inspecting, because those elements say which wording the
  model cannot use.
- **An element labelled with fewer than three descriptions contributes fewer rows**, rather than a
  clamped duplicate of `name` filed under `intent` — which would put identical predictions on both
  sides of `agreement` and measure a gap in the dataset instead of the model. The run logs how many
  pairs it skipped.

A multi-phrasing run's filename carries `-3phrasing-`, because its headline sits below the same
model's `name`-only result for a reason that has nothing to do with model quality.

**Prompt caching is not comparable across the two.** The phrasings of one element run adjacently, so
the image stays in the server's prompt cache and `cached_input_fraction` comes out higher than in a
single-phrasing run. Compare models on total input tokens, never on that fraction.

---

## Image size and context limits

**Screenshots are sent at their original resolution.** Downscaling to 1080 used to be the default;
that was the wrong trade here, because the benchmark exists to measure how well a model can ground an
element and shrinking the screenshot first only handicaps it. `--max-image-dim 1080` still answers the
separate question of what a downscaling consumer would see.

**Downscaling is not a benchmark setting.** A comparison across models is only meaningful when every
model saw the screenshot the consumer will send it; a run at any non-zero value is answering a
different question and cannot be put beside the others.

Two consequences worth knowing.

**Context.** A 1080×2400 screenshot becomes ~3,300 image tokens against ~900 at max-dim 1080 — enough
on its own to overflow a small context and produce

```
the request exceeds the available context size (4096 tokens), try increasing it
```

That comes from the model server, not from this code. Raise the server's own context (`--ctx-size` on
llama.cpp, `--max-model-len` on vLLM) or pass a smaller `--max-image-dim`.

**Cost.** Native multiplies input tokens per element by roughly four, so results either side of this
setting are not comparable on token counts. Measured on one run: a 1080×2400 screenshot cost 3,545
input tokens per element, a 716×955 one 1,076, a 387×862 one 626. The image is essentially the whole
input — the element description moves it by 0–2 tokens, correlating at r = −0.01. `max_image_dim` is
recorded in every result file for exactly this reason.

Neither changes what is measured: predictions and ground truth are both compared in normalised
coordinates. But `_load_image_b64` must return the *resized* dimensions when it does resize, since
pixel answers are normalised against them.

---

## Response parsing

Models disagree wildly on output format, so `_parse_response()` in `scoring/parsing.py` handles
11 shapes: clean
`[x1,y1,x2,y2]`, nested arrays, JSON-encoded strings, markdown fences, `{x,y,width,height}` dicts,
list-wrapped values, centre-point-only `{x,y}`, GUI-Owl's native `{"x": [cx, cy]}`, multiple JSON
objects, 5+ numbers (sliding window of 4), and preamble prose (regex fallback). Malformed JSON gets
one repair attempt via `json-repair`.

`_try_bbox_from_nums(nums, img_w, img_h)` uses a two-pass sliding window: pass 1 prefers a window
where all four raw values are already ≤ 1.0, pass 2 accepts pixel-scale values and divides by the
image dimensions.

**If a new model returns something none of those cover, extend `_parse_response` in
`scoring/parsing.py`.**

### Temperature: greedy here, absent on a frontier model

`temperature=0` is sent by default, and it is load-bearing rather than decorative. The main targets
are self-hosted llama.cpp and vLLM endpoints, which honour it, and greedy decoding is what makes a
coordinate reproducible across runs. **llama.cpp's own default is 0.8**, so simply not sending it
would put sampling noise straight into digit tokens — where one flip turns `x=432` into `x=332`, a
100px error rather than a rephrasing.

Frontier models refuse it: GPT-5.x answers *"Unsupported value: 'temperature' does not support 0.0
with this model"*, and recent Claude models return 400 for any non-default value. Two ways out, and
the second needs no foresight:

- `--temperature -1` sends none at all.
- The script detects the refusal itself — a 400 or 422 whose body mentions `temperature` — drops the
  parameter **for the rest of the run**, and retries that element inline. One wasted round-trip for
  the whole run, not one per element. The match is on the body, so a 400 about a malformed image or a
  context overflow is not mistaken for this.

Either way `temperature` and `temperature_dropped` are recorded, because they decide whether two runs
are strictly comparable: under greedy they reproduce, under the server's own default they need not.

### Prose is a failure, not a bbox

The last of those shapes — preamble prose followed by coordinates — used to be handled by scraping
*every* number out of the text and sliding a window of four across them. A model that reasons in a
numbered list and never commits to coordinates therefore handed over `1. … 2. … 3. … 4. …` and got a
bounding box back **with no error**: 34 identical copies of one such box in a single run, 23% of its
predictions carrying the fingerprint.

Two rules now separate a real answer from a manufactured one:

- **Numbers must arrive as a group.** `_numeric_runs` breaks a run wherever more than three letters
  separate two numbers, so `[0.24, 0.81, 0.83, 0.90]` and `x1=24, y1=345, x2=233, y2=410` are single
  groups while a sentence between two numbers is not. **Size cannot be the test:** the artefact box
  was 0.00411 wide and the smallest real element in the dataset is 0.00370, so any floor rejecting
  the fabrication also rejects genuine 4px icons.
- **One scale decision per box, not per value.** `_normalize_val` divides by the image dimension only
  when a value exceeds 1.0, so in `[1, 2, 3, 4]` the `1` was read as an already-normalised 1.0 — the
  right-hand edge of the screen — while 2, 3 and 4 were read as pixels. `_normalize_window` decides
  once from the largest value, which leaves a legitimate edge-hugging `[0, 812, 300, 906]` intact.

An answer with no group of numbers now returns `no coordinates in response — the model answered in
prose: …`.

### Truncation is a failure too

`max_tokens` used to be 512. A reasoning model spends its output budget *before* answering, so
roughly half of all calls to one 12B thinking model hit `finish_reason="length"` and returned
`content=""`. The caller then fell back to `reasoning_content` and handed raw reasoning prose to the
parser, which scraped the `1.` and `2.` of a numbered list into a box **and reported no error**. One
bogus box, repeated, is what a 100-element run recorded as 16% centroid accuracy — a number that
measured the token cap, not the model.

`max_tokens` is now 2048 and `finish_reason == "length"` returns an explicit truncation error before
any fallback runs. **The general lesson: a component that silently manufactures a plausible answer
instead of failing turns a config problem into corrupt data.**

### Thinking models

`--thinking-budget 0` is meant to disable chain-of-thought. The value is sent as `budget_tokens` only
when `>= 0`; the default `-1` leaves the model's own default.

**`budget_tokens` is a vLLM/Anthropic-shaped parameter and llama.cpp ignores it**, so against a
llama.cpp endpoint the flag is a no-op — verified on one 12B thinking model, output tokens stayed at
430–512 with `budget_tokens=0`. Worse, on that endpoint passing it took 24.4s instead of 19.4s, hit
the token cap and returned empty content, which the benchmark reports as an error. **Use it only
against vLLM.**

Thinking models are also slow enough to change the plan: ~20s per element on a single-slot server
against 2.9s for Qwen2.5-VL-7B, so a full 10,307-element run is roughly **56 hours** rather than under
three. Either run it through vLLM where thinking can actually be capped, or use `--limit` and report
the sample size.

---

## Output JSON format

```json
{
  "model": "qwen2.5-vl",
  "date": "2026-08-01T22:10:03+00:00",
  "dataset_path": "/abs/path/dataset-v1.jsonl",
  "description_indices": [0, 1, 2],
  "metric": "centroid",
  "workers": 5,
  "max_tokens": 2048,
  "max_image_dim": 0,
  "coord_grid": 0,
  "prompt_style": "pixels",
  "temperature": 0,
  "temperature_dropped": false,
  "thinking_budget": -1,
  "served_model": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
  "stopped_early": false,
  "summary": {
    "total_elements": 30921,
    "unique_elements": 10307,
    "selected_elements": 30921, "completed_fraction": 1.0,
    "metric": "centroid",
    "primary_accuracy": 0.8519,
    "by_description": {
      "0": {"style": "name",   "total": 10307, "primary_accuracy": 0.8440,
            "iou_accuracy": 0.4635, "mean_iou": 0.4621,
            "avg_input_tokens": 3545.8, "avg_output_tokens": 19.7,
            "avg_latency_ms": 1250.0, "error_count": 0},
      "1": {"style": "label",  "primary_accuracy": 0.8811},
      "2": {"style": "intent", "primary_accuracy": 0.8306}
    },
    "agreement": {
      "phrasings": 3, "scored_elements": 10307,
      "any_pass_count": "...", "any_accuracy": "...",
      "all_pass_count": "...", "all_accuracy": "...",
      "mixed_count": "..."
    },
    "iou_pass_count": 15211, "iou_accuracy": 0.4919,
    "mean_iou": 0.4621, "median_iou": 0.4931, "iou_threshold": 0.5,
    "centroid_pass_count": 26342, "centroid_accuracy": 0.8519,
    "point_only_count": 0,
    "bbox_count": 30921, "bbox_coverage": 1.0, "iou_accuracy_given_bbox": 0.4919,
    "clamped_pred_count": 315,
    "scale_check": {"predictions": 30921,
                    "pred_median_cx": 0.499, "pred_median_cy": 0.547,
                    "gt_median_cx": 0.500,  "gt_median_cy": 0.5575,
                    "x_ratio": 0.998, "y_ratio": 0.9812,
                    "axes_disagree": false, "suspect": false},
    "offset_check": {"predictions": 30921,
                     "median_dx": 0.0046, "median_dy": 0.0004,
                     "median_abs_dx": 0.006, "median_abs_dy": 0.0033,
                     "boxes": 30921,
                     "median_width_ratio": 0.913, "median_height_ratio": 0.8182},
    "total_input_tokens": 109799005, "total_output_tokens": 607624,
    "avg_input_tokens": 3551.0, "avg_output_tokens": 19.7,
    "total_cached_input_tokens": 16386, "cached_input_fraction": 0.2213,
    "p95_output_tokens": 23,
    "elements_with_token_counts": 30921,
    "error_count": 0, "timeout_count": 0,
    "avg_latency_ms": 1250.0, "p50_latency_ms": 1100.0, "p95_latency_ms": 2800.0
  },
  "results": [
    {
      "screenshot_id": "sample-screen-1",
      "element_id": "e3",
      "description": "the login button",
      "description_index": 0,
      "description_style": "name",
      "gt_bbox":   {"x": 0.12, "y": 0.45, "width": 0.20, "height": 0.06},
      "pred_bbox": {"x": 0.11, "y": 0.44, "width": 0.21, "height": 0.07},
      "pred_point": null,
      "iou": 0.823,
      "pass_iou": true,
      "pass_centroid": true,
      "error": "",
      "latency_ms": 1140.0,
      "input_tokens": 712,
      "output_tokens": 19,
      "cached_input_tokens": 172,
      "img_w": 1080,
      "img_h": 2400,
      "raw": "[0.11, 0.44, 0.32, 0.51]"
    }
  ]
}
```

`pass_centroid` is the centroid metric. `pred_point` is set instead of `pred_bbox` when the model
returned a bare click point.

**`click_inside` is gone**, at row level and in the summary. It was the original name for the
centroid metric, kept as an alias after the rename; every reader now takes `pass_centroid` /
`centroid_accuracy`, which files written under the old name also carry, so an older result still
reads fine. `--rescore` strips it from every row it emits on every path, since carrying it forward
would leave it holding the *previous* verdict — a stale field contradicting `pass_centroid` in the
same row is worse than an absent one.

Two per-row fields exist so a result can be re-scored later without the images and without guessing:

- **`img_w` / `img_h`** — the dimensions this answer was normalised against, which are the *resized*
  ones when `--max-image-dim` was in force, not the file's. `--rescore` reads them first; re-deriving
  them from the image is silently wrong for a downscaled run.
- **`raw`** — **not truncated** (`RAW_RESPONSE_CHARS = 0`). It is the only field that can tell a
  model's mistake from the parser's, and every coordinate-convention bug here has been diagnosed by
  reading it: GUI-Owl's 0–1000 grid was found by re-parsing `raw` out of a finished file rather than by
  re-running 30,921 calls. The size is bounded by `--max-tokens`, ~8 KB per row at the 2048 default,
  which in practice is 27 MB for a full run where Qwen writes ~25 characters per answer.

### The six fields to read before believing an accuracy

- **`bbox_coverage`** — the share of requests that returned a bounding box at all, and the first thing
  to read when comparing models for a consumer that needs one. It is a **gate**: GUI-Owl at 15.1% and
  Qwen2.5-VL at 100% are not answering the same question, so their centroids are not a ranking.
- **`scale_check`** — whether predictions are distributed like the ground truth on each axis. The only
  guard that works in aggregate, and therefore the only one that can see a coordinate-convention error
  at all. `suspect: true` means the accuracy beside it is a property of the harness.
- **`offset_check`** — which *direction* the model is wrong in. `median_dx`/`median_dy` are signed, in
  screen fractions, with `median_abs_*` beside them and `median_width_ratio`/`median_height_ratio`
  after. It separates three faults that all arrive as one low accuracy: **displacement** (signed
  medians far from 0 — a fixed aim error), **scatter** (signed near 0 while absolute is large — the
  model is not finding the element at all), and **mis-sizing** (position fine, ratio off — which is
  what turns "centroid passes, IoU fails" into a cause).

  **`scale_check` is structurally blind to the first of those**, because shifting a whole
  distribution barely moves a ratio of medians. Measured: a frontier model on the float prompt sat
  0.0988 of the screen too low on *every* prediction while `scale_check` reported `y_ratio` 0.857 —
  inside its 0.8–1.25 band, `suspect: false`. This is the block that turns "that model scores 0%"
  into a stateable model weakness.
- **`clamped_pred_count`** — predictions pushed against the [0,1] edge. This catches *prose being
  scraped into a box*, where the parser salvages numbers out of a sentence and lands on a full-frame
  rectangle. A **high** value means the number below it measures the harness, not the model. A low
  one means nothing on its own: a full-width list row genuinely has width 1.0.

  It does **not** catch a wrong coordinate scale, and an earlier version of this file claimed it did.
  That is false, and GUI-Owl disproved it: it answers on a 0–1000 grid, was read as 1080×2400 pixels,
  and clamped 0.6% of 30,921 predictions — indistinguishable from a clean run. Dividing by a number
  *larger* than the true divisor moves every value toward zero, not toward 1.0; `916 / 1080` is 0.85
  and clamps nothing. Use `scale_check` for that, which is the field written after this one failed to
  fire.
- **`workers`** — every latency figure was measured at this concurrency, so two runs at different
  values are not comparable on latency. On a single-slot server extra workers add queue wait to each
  request rather than throughput.
- **`elements_with_token_counts`** — kept separate from the token totals because a server that reports
  no `usage` block leaves them at 0, which is not the same finding as a model that is genuinely cheap.

**`cached_input_fraction` is recorded but should not be used to compare models.** Prompt caching is a
property of the server's recent history, not of the model: it survives across runs, so the same
benchmark twice reports different figures. It is there for diagnosing why input tokens or latency
moved.
