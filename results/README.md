# Published results

One approved result per model and approach, the summaries the README's tables are read from:

```
results/
  vision/   claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b  gui-owl-1-5-8b    the screenshot track
  xml/      claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b                    the tree track
  cascade/  claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b                    tree first, screenshot as the fallback
```

Each file is the `summary` of a full run with its per-request rows omitted, a `note` saying what it
is, and a `config` block: the configuration under [`config/`](../config/) it corresponds to, by name
and hash. Dates, prompt hashes, endpoints and token totals are inside the files, not in their names.
Every figure is over the 10,568 elements with `annotation.origin = gpt-5.6` (see
[`data/README.md`](../data/README.md)); the two vision files that were run over the whole ground
truth say so in their note and carry that population as `summary.by_source.gpt`.

A new run of the same config lands in `benchmark-results/` (gitignored) with the date in its name;
a file comes here only when it replaces the published figure for that model and track.

## From the screenshot

|                             | tap-point correct (centroid) | box correct (IoU ≥ 0.5) | median IoU | returned a box | asked for |
|-----------------------------|------------------------------|-------------------------|------------|----------------|-----------|
| **GPT-5.6**                 | **96.2%**                    | **81.7%**               | 0.890      | 100%           | pixels    |
| **Claude Opus 4.8**         | 91.3%                        | 61.2%                   | 0.617      | 100%           | pixels    |
| **Qwen2.5-VL-7B-Instruct**  | 84.6%                        | 55.4%                   | 0.553      | 99.98%         | pixels    |
| **GUI-Owl-1.5-8B-Instruct** | 81.8%                        | 2.1%                    | 0.216      | 16.4%          | floats    |

Every row is a benchmark answer: screenshot + one description in, box out, under the `name`
phrasing. All three phrasings were asked; averaged over the three, GPT-5.6 scores 96.10% centroid
with 93.05% of elements passing under all three, Opus 91.97% and 87.07%, Qwen 85.99% and 75.58%
(see [How much the wording matters](#how-much-the-wording-matters)). In a result file this table is
`summary.by_source.gpt`; a run over the whole file also reports the all-elements figure (Qwen: 83.5%
centroid over 11,914).

*Deployment:* the Qwen row was measured against a production vLLM serving on a GPU cluster, the
GUI-Owl row against llama.cpp serving the Q4_K_M GGUF on an Apple Silicon Mac mini, single-stream.
Accuracy is comparable across the two; latency is not, and is not quoted here.

**Mind the last column.** GPT-5.6, Opus and Qwen were asked for integer pixels, GUI-Owl for floats
in [0, 1], and that is not a free choice — see
[How you ask for the coordinates changes the score](../docs/methodology.md#--prompt-style--the-coordinate-format-is-a-model-selection-variable-not-a-detail),
where the same Opus model scores 7.79% or 90.33% on one 200-element set depending on nothing but
that. Qwen ignores the instruction and answers in pixels regardless, so its figure is unaffected;
GUI-Owl answers on a 0–1000 grid and has never been measured under the pixel prompt.

**[Browse every row screenshot by screenshot →](https://dataset-review-ui-test.kobiton.com/compare)**
the ground-truth boxes and the model's side by side over each screenshot, every prediction tagged
with its IoU and colour-coded by band, and a per-element list with the IoU and centroid verdicts.

## From the accessibility tree

The same 10,568 elements, the same three phrasings, no image: the filtered page source and one
description go to the model, one XPath or `NOT_FOUND` comes back, the XPath is run on the full
tree and the first node it selects is graded by the same centroid rule. Over all three phrasings —
31,704 requests per model, percentages of that total:

|                                              |        GPT-5.6 | Claude Opus 4.8 |   Qwen2.5-VL-7B |
|----------------------------------------------|----------------|-----------------|-----------------|
| answered, right element                      | 58.1% (18,433) |  57.2% (18,125) |  35.8% (11,339) |
| answered, wrong element                      |  10.5% (3,327) |    9.5% (2,999) |    9.8% (3,098) |
| not answered                                 |  31.4% (9,944) |  33.4% (10,580) |  54.4% (17,261) |
| ↳ of which `NOT_FOUND`                       |          9,637 |          10,251 |          12,757 |
| ↳ of which the XPath did not compile         |              3 |              46 |             160 |
| ↳ of which the XPath matched no node         |            304 |             283 |           4,344 |
| right among the answered                     |          84.7% |           85.8% |           78.5% |
| elements right under **all three** phrasings |          54.6% |           52.9% |           23.4% |

Two models that are seven points apart on the screenshot are within a point of each other on the
tree, in every row; the third, seven points behind Opus on the screenshot, is 21 points behind it
here. For GPT-5.6 and Opus the ceiling is the data's, not the model's:

- **A third of the requests get no answer, and half of those are the keyboard.** 5,060 of the
  31,704 requests describe a key on the on-screen keyboard; the page source holds the application
  window only, so no such node exists, and GPT-5.6 and Opus say `NOT_FOUND` on ~96% of them (Qwen
  on 90%). Excluding those rows, *not answered* is 19.2% for GPT-5.6 and 21.3% for Opus over the
  remaining 26,644, and 47.4% for Qwen. The rest are interfaces drawn without a usable tree, such as
  one container for a whole screen, links inside a text run — which no prompt can fix.
- **When the tree does name the element, the model is right about 85% of the time**, and the ~10%
  it gets wrong is the part no fallback recovers: see the cascade. The phrasing barely moves the
  right-answer rate for GPT-5.6 and Opus (57–58% under all three); `intent` draws a few more wrong
  answers.
- **Qwen2.5-VL: close from the screenshot, far behind from the tree.** From the screenshot it is
  6.7 points behind Opus (84.6% against 91.3%); from the tree it is 21 points behind (35.8% against
  57.2%), answers right a third of the time, and moves with the wording (40% under `name`, 33% under
  `label`). 4,344 of its XPaths (13.7% of requests) compile but select nothing, fourteen times
  GPT-5.6's 304: they name attributes the tree does not have. On 6 further requests, left out of the
  table, it repeated one path step until the 2,048-token cap; GPT-5.6 and Opus produced no such
  answer in 31,704 (see [the error analysis](../docs/error-analysis.md#tree-track-answers-stuck-in-a-repetition-loop)).

Every row's `xml_outcome` and the per-file counts (`summary.xml_outcomes`) are in the result files;
[the methodology page](../docs/methodology.md#--input-xml--the-xml-track) explains what each outcome
means and how the prompt is handled. **The prompt these three were measured with is not the one in
the package**: it is a production locator's template, and the files record its hash
(`xml_prompt_sha256`); a run with the shipped prompt will differ by the prompt.

## Tree first, screenshot as the fallback

A locator in the field would try the tree first and show the model the screenshot only when the
tree yields nothing. No new request is needed to measure that: the two tracks above are joined per
(element, phrasing) by `ai-testing-benchmark cascade`. Every request falls into exactly one of lines
1, 2, 4, 5 or 6; lines 3, 7 and 8 are sums. Over all three phrasings, 31,704 requests per model
(31,698 for Qwen, whose 6 tree errors are left out), percentages of that total:

| # |                                              |            GPT-5.6 |    Claude Opus 4.8 |      Qwen2.5-VL-7B |
|---|----------------------------------------------|--------------------|--------------------|--------------------|
| 1 | tree answered correctly                      |     58.1% (18,433) |     57.2% (18,125) |     35.8% (11,339) |
| 2 | tree answered incorrectly                    |      10.5% (3,327) |       9.5% (2,999) |       9.8% (3,098) |
| 3 | tree did not answer → sent to the screenshot |      31.4% (9,944) |     33.4% (10,580) |     54.4% (17,261) |
| 4 | screenshot answered correctly                |      30.5% (9,675) |      30.2% (9,575) |     45.9% (14,555) |
| 5 | screenshot answered incorrectly              |         0.9% (269) |       3.2% (1,005) |       8.5% (2,702) |
| 6 | screenshot did not answer                    |             0% (0) |             0% (0) |           0.0% (4) |
| 7 | **correct overall** (1 + 4)                  | **88.7% (28,108)** | **87.4% (27,700)** | **81.7% (25,894)** |
| 8 | **incorrect overall** (2 + 5 + 6)            |      11.3% (3,596) |     12.6% (4,004) |     18.3% (5,804) |

By phrasing, line 7 is 89.2% / 90.5% / 86.3% for GPT-5.6, 86.8% / 89.4% / 85.9% for Opus and
81.5% / 87.4% / 76.2% for Qwen (`name` / `label` / `intent`): the same ordering as on the
screenshot. All eight lines per phrasing are in the details block at the end of this section.

How to read it:

- **A wrong XPath is final.** The fallback fires on *not answered*, never on *answered wrongly*, so
  line 2 is lost for good and the cascade can score below the screenshot alone. It does here for
  every model: GPT-5.6's screenshot answers alone are right on 96.1% of these same 31,704 requests,
  the cascade on 88.7%; Opus 92.0% against 87.4%; Qwen 86.0% against 81.7%. What line 2 loses
  (10.5% / 9.5% / 9.8%) outweighs what the screenshot gets wrong in line 5 (0.9% / 3.2% / 8.5%).
- **The screenshot answers come from each model's full vision run**, graded by the same centroid
  rule as the screenshot table above (`pixels` prompt); the cascade uses them only on the rows the
  tree did not answer. On those rows the screenshot is right 97.3% of the time for GPT-5.6, 90.5%
  for Opus and 84.3% for Qwen — the rows the tree could not name are not hard from the screenshot.
- **Line 6 is 0 by construction, and that is the right reading for this dataset.** Every element
  the cascade asks about is on the screen, so it is in the human ground truth; "not found" is never
  the correct answer here, and letting the model decline could only turn a request in line 4 or 5
  into a miss. The screenshot prompt therefore allows no "not found" answer, and line 6 holds only
  parse failures and transport errors: none for GPT-5.6 and Opus, and for Qwen the 4 requests it
  answered in prose with no box. What a model does when the element is genuinely absent is a
  different question, and needs its own set of absent-element queries to measure.
- **The keyboard sits in line 3.** The ~4,850 keyboard requests GPT-5.6 and Opus could not answer
  from the tree (4,643 for Qwen) all go to the screenshot, where they are handled like any other
  control. The cascade therefore recovers the data gap, and lines 4–5 should be read with that in
  mind.

The sublines of 3 and 6 and the JSON behind the table are in [`cascade/`](cascade/) and described
in [the methodology page](../docs/methodology.md#cascade--tree-first-screenshot-as-the-fallback).

<details>
<summary><b>Details of the same eight lines per description phrasing</b></summary>

Each phrasing is 10,568 requests; the "all" column is the table above.

**GPT-5.6**

| # |                                              | all 3 phrasings (31,704) |     name (10,568) |    label (10,568) |   intent (10,568) |
|---|----------------------------------------------|--------------------------|-------------------|-------------------|-------------------|
| 1 | tree answered correctly                      |           58.1% (18,433) |     58.1% (6,145) |     58.4% (6,170) |     57.9% (6,118) |
| 2 | tree answered incorrectly                    |            10.5% (3,327) |     10.0% (1,054) |        8.6% (911) |     12.9% (1,362) |
| 3 | tree did not answer → sent to the screenshot |            31.4% (9,944) |     31.9% (3,369) |     33.0% (3,487) |     29.2% (3,088) |
| 4 | screenshot answered correctly                |            30.5% (9,675) |     31.0% (3,279) |     32.1% (3,397) |     28.4% (2,999) |
| 5 | screenshot answered incorrectly              |               0.9% (269) |         0.9% (90) |         0.9% (90) |         0.8% (89) |
| 6 | screenshot did not answer                    |                 0.0% (0) |          0.0% (0) |          0.0% (0) |          0.0% (0) |
| 7 | **correct overall** (1 + 4)                  |       **88.7% (28,108)** | **89.2% (9,424)** | **90.5% (9,567)** | **86.3% (9,117)** |
| 8 | **incorrect overall** (2 + 5 + 6)            |            11.3% (3,596) |     10.8% (1,144) |      9.5% (1,001) |     13.7% (1,451) |

**Claude Opus 4.8**

| # |                                              | all 3 phrasings (31,704) |     name (10,568) |    label (10,568) |   intent (10,568) |
|---|----------------------------------------------|--------------------------|-------------------|-------------------|-------------------|
| 1 | tree answered correctly                      |           57.2% (18,125) |     57.2% (6,041) |     57.2% (6,043) |     57.2% (6,041) |
| 2 | tree answered incorrectly                    |             9.5% (2,999) |      9.5% (1,003) |        7.9% (836) |     11.0% (1,160) |
| 3 | tree did not answer → sent to the screenshot |           33.4% (10,580) |     33.4% (3,524) |     34.9% (3,689) |     31.9% (3,367) |
| 4 | screenshot answered correctly                |            30.2% (9,575) |     29.6% (3,129) |     32.2% (3,405) |     28.8% (3,041) |
| 5 | screenshot answered incorrectly              |             3.2% (1,005) |        3.7% (395) |        2.7% (284) |        3.1% (326) |
| 6 | screenshot did not answer                    |                 0.0% (0) |          0.0% (0) |          0.0% (0) |          0.0% (0) |
| 7 | **correct overall** (1 + 4)                  |       **87.4% (27,700)** | **86.8% (9,170)** | **89.4% (9,448)** | **85.9% (9,082)** |
| 8 | **incorrect overall** (2 + 5 + 6)            |            12.6% (4,004) |     13.2% (1,398) |     10.6% (1,120) |     14.1% (1,486) |

**Qwen2.5-VL-7B** — the 6 tree errors (2 `name`, 4 `label`) are left out of the counts.

| # |                                              | all 3 phrasings (31,698) |     name (10,566) |    label (10,564) |   intent (10,568) |
|---|----------------------------------------------|--------------------------|-------------------|-------------------|-------------------|
| 1 | tree answered correctly                      |           35.8% (11,339) |     39.7% (4,195) |     32.7% (3,453) |     34.9% (3,691) |
| 2 | tree answered incorrectly                    |             9.8% (3,098) |        9.0% (951) |        5.9% (627) |     14.4% (1,520) |
| 3 | tree did not answer → sent to the screenshot |           54.4% (17,261) |     51.3% (5,420) |     61.4% (6,484) |     50.7% (5,357) |
| 4 | screenshot answered correctly                |           45.9% (14,555) |     41.8% (4,413) |     54.7% (5,778) |     41.3% (4,364) |
| 5 | screenshot answered incorrectly              |             8.5% (2,702) |      9.5% (1,005) |        6.7% (706) |        9.4% (991) |
| 6 | screenshot did not answer                    |                 0.0% (4) |          0.0% (2) |          0.0% (0) |          0.0% (2) |
| 7 | **correct overall** (1 + 4)                  |       **81.7% (25,894)** | **81.5% (8,608)** | **87.4% (9,231)** | **76.2% (8,055)** |
| 8 | **incorrect overall** (2 + 5 + 6)            |            18.3% (5,804) |     18.5% (1,958) |     12.6% (1,333) |     23.8% (2,513) |

</details>

## How much the wording matters

Every element carries three descriptions — `name`, `label`, `intent` — and every model was asked
all three. The screenshot table shows `name`; this section is what the other two add. All four
columns are on the human ground truth, over the same 10,568 elements as the screenshot table.

**Note:** GUI-Owl's centroid and its IoU are answers to two different questions. It is an agentic
model trained to emit a click point `(x, y)`, so 84.6% of its answers carry no box at all, and only
17.9% of the boxes it does return reach IoU ≥ 0.5. A point has no area, so it scores zero on IoU by
construction. If what you need is a bounding box, coverage is a **gate**, not a metric, and those
two are not on the same leaderboard:

```
iou_accuracy = bbox_coverage × iou_accuracy_given_bbox
     2.75%    =     15.4%     ×        17.9%
```

Asked the same element three ways, every model moves more than the gap between models:

| Phrasing                   | example                                                    | GPT-5.6    | Opus 4.8   | Qwen2.5-VL | GUI-Owl    |
|----------------------------|------------------------------------------------------------|------------|------------|------------|------------|
| `name` — short common name | *the scan QR code button*                                  | 96.19%     | 91.26%     | 84.60%     | 81.76%     |
| `label` — structural       | *the blue button with the text scan QR code*               | **96.56%** | **93.37%** | **90.17%** | **87.31%** |
| `intent` — functional      | *the button that lets the user scan their sign-in QR code* | 95.56%     | 91.29%     | 83.19%     | 84.63%     |

All four find `label` easiest, it is the phrasing that repeats the element's visible text most
often. The hardest phrasing splits two and two: GPT-5.6 and Qwen on `intent`, Opus and GUI-Owl on
`name` — Opus by a hair (91.26% against 91.29%), GUI-Owl by three points.

On the tree the wording hardly matters to the right-answer rate for GPT-5.6 and Opus — 57–58%
under every phrasing — because what decides a tree request is whether the element has a node at
all, not how it was described. Qwen is the exception: 40% / 33% / 35% under `name` / `label` /
`intent`. What moves is the wrong-answer rate: `intent`, the phrasing with no text handle, draws
12.9% wrong from GPT-5.6 against 8.6% under `label`.

A single-phrasing headline flatters a model, because `name` is usually the element's visible text
and `intent` deliberately carries no text handle at all. Scoring all three is what makes the figure
describe the ways a tester might actually phrase a locator rather than the easiest one.

**And the average still flatters it.** Averaging the three counts an element as 2-of-3 correct;
what a test suite needs is the element working *whatever* the tester typed. Measured per element,
on the screenshot:

|                         | average centroid | passes under **all three** |
|-------------------------|------------------|----------------------------|
| GPT-5.6                 | 96.10%           | **93.05%**                 |
| Claude Opus 4.8         | 91.97%           | **87.07%**                 |
| Qwen2.5-VL-7B-Instruct  | 85.99%           | **75.58%**                 |
| GUI-Owl-1.5-8B-Instruct | 84.57%           | **72.30%**                 |

So the figure to plan reliability against is three points below the headline for the strongest
model here, five for Opus, and ten to twelve for the open-weight ones — the gap widens as the model
weakens, which is the opposite of what an average suggests. (`summary.agreement` in every
multi-phrasing result file carries these counts; on the tree it is 54.6%, 52.9% and 23.4%, in the
table above.)

## Reading a result file

A run prints its headline and writes the JSON:

```
=== qwen2.5-vl (primary metric: centroid) ===
  Centroid Accuracy (center inside gt): 85.2%  (26,342/30,921 pass) *
  IoU Accuracy  (IoU≥0.5): 49.2%
  Mean IoU:     0.462
  Median IoU:   0.493
  Errors:       0 (timeout: 0)
```

The JSON carries far more: `config` is the configuration the run used; `summary.by_description`
has each phrasing scored separately; `summary.agreement` has how often the *same* element passes
under one / all / some wordings; `summary.by_source` has the figures split by which labelling the
element came from (`gpt-` / `opus-` ids); `summary.answer_classes` and each row's `answer_class`
say what the model did when it was wrong (see [the error analysis](../docs/error-analysis.md)); and
every row keeps the model's own untruncated answer in `raw` plus the `img_w` / `img_h` it was
normalised against. A tree-track file (`xml-…`) adds `summary.xml_outcomes`, each row's `xpath` and
`xml_outcome`, and the prompt's name and hash; `ai-testing-benchmark cascade --json` writes the
eight cascade lines with their sublines, overall and per phrasing.
[The methodology page](../docs/methodology.md#output-json-format) documents every field.

**Read `served_model` before quoting a number.** `--model` is a request, not a guarantee: a server
holding a single weight file answers every request with whatever is loaded, whatever name you ask
for. Each run therefore records who actually replied, warns when that disagrees with the request,
and marks the filename `-SERVED-<name>`.

Latency is recorded per run, but each figure describes its own serving setup — it is not comparable
across rows, and not a production figure for any of them.
