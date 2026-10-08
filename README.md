# AI Testing Benchmark

**Can a model find a UI element from a plain-English description using the screenshot, the screen's accessibility tree, or tree first with the screenshot as the fallback?**

If it can, UI tests may not need brittle selectors like:

```java
findElement(By.xpath("//android.widget.Button[@resource-id='btn_scan_qr']"))
```

Instead, a test can describe the target element in plain English:

```java
findElement(byDescription("the scan QR code button"))   // illustrative
```

and let a model locate it, either visually from the screenshot or structurally from the page source.

This repository benchmarks both, and the combination a locator would actually run. 
The **screenshot track** shows a vision model the screen and one description and asks for a box. 
The **tree track** shows a text model the filtered accessibility tree and the same description and asks for one XPath or `NOT_FOUND`;
the XPath is run on the tree and the node it selects is the answer. 
A third scorer joins the two into the **cascade**: tree first, screenshot only where the tree gave nothing. 
All three are scored on the same elements, the same descriptions and the same ground truth.

It contains **841 real Android screenshots**, each with its page source, labeled three times - by two frontier models independently 
and by a human adjudicating between them, and **10,568 human-verified elements** in the benchmark population, each described in three different ways. 
The runner works with any **OpenAI-compatible endpoint**, including vLLM, llama.cpp, Ollama, and hosted APIs.

It exists because we needed to choose a self-hosted model and a locator strategy, and wanted a benchmark tailored to the question we actually needed to answer. 
Published so you can check our numbers and score your own model on the same corpus.

---

## Results

**Scored against the human-adjudicated ground truth** (`data/dataset-v1-human.jsonl`): an independent human annotator 
settled every element the two frontier labellings disagreed on and verified the ones they agreed on 
(how it was built is under [The dataset](#the-dataset)). 
Every figure below is over the same **10,568 elements** - the ones GPT-5.6 described, which every model here has an answer for - 
under the `name` phrasing unless a table says otherwise.

### At a glance

Three ways to locate the same element, three models that were measured under all three. 
Correct means that the center of the returned box, or of the node selected by the XPath, falls within the human-annotated box for that element.

|                        | Screenshot alone          | Tree alone | Tree first, screenshot as the fallback |
|------------------------|---------------------------|------------|----------------------------------------|
| **GPT-5.6**            | 96.2%                     | 58.1%      | 89.2%                                  |
| **Claude Opus 4.8**    | 91.3%                     | 57.2%      | 86.8%                                  |
| **Qwen2.5-VL-7B**      | 84.6%                     | 35.8%      | 81.5%                                  |

Tree alone and the cascade are not the same measurement as the screenshot alone, and the gap between the columns is the point:
a third of the elements have no node the tree can name (the keyboard, and interfaces drawn without a tree - see [On the accessibility tree](#on-the-accessibility-tree)),
and a wrong XPath is final, the fallback never fires on it. The three subsections below give each column in full.

### From the screenshot

|                             | tap-point correct (centroid) | box correct (IoU ≥ 0.5) | median IoU | returned a box | asked for |
|-----------------------------|------------------------------|-------------------------|------------|----------------|-----------|
| **GPT-5.6**                 | **96.2%**                    | **81.7%**               | 0.890      | 100%           | pixels    |
| **Claude Opus 4.8**         | 91.3%                        | 61.2%                   | 0.617      | 100%           | pixels    |
| **Qwen2.5-VL-7B-Instruct**  | 84.6%                        | 55.4%                   | 0.553      | 99.98%         | pixels    |
| **GUI-Owl-1.5-8B-Instruct** | 81.8%                        | 2.1%                    | 0.216      | 16.4%          | floats    |

Every row is a benchmark answer: screenshot + one description in, box out.
All three phrasings were asked and `name` is shown; averaged over the three,
GPT-5.6 scores 96.10% centroid with 93.05% of elements passing under all three,
Opus 91.97% and 87.07%, Qwen 85.99% and 75.58% (see [How much the wording matters](#how-much-the-wording-matters)). 
In a result file this table is `summary.by_source.gpt`; a run over the whole file also reports the all-elements figure 
(Qwen: 83.5% centroid over 11,914).

*Deployment:* the Qwen row was measured against a production vLLM serving on a GPU cluster, 
the GUI-Owl row against llama.cpp serving the Q4_K_M GGUF on an Apple Silicon Mac mini, single-stream. 
Accuracy is comparable across the two; latency is not, and is not quoted here.

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

### From the accessibility tree

The same 10,568 elements, the same three phrasings, no image: the filtered page source and one description go to the model,
one XPath or `NOT_FOUND` comes back, the XPath is run on the full tree and the first node it selects is graded by the same
centroid rule. Over all three phrasings — 31,704 requests per model, percentages of that total:

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

Two models that are seven points apart on the screenshot are within a point of each other on the tree, in every row;
the third, seven points behind Opus on the screenshot, is 21 points behind it here. For GPT-5.6 and Opus the ceiling is the data's, not the model's:

- **A third of the requests get no answer, and half of those are the keyboard.** 5,060 of the 31,704 requests describe a key on
  the on-screen keyboard; the page source holds the application window only, so no such node exists, and GPT-5.6 and Opus say
  `NOT_FOUND` on ~96% of them (Qwen on 90%). Excluding those rows, *not answered* is 19.2% for GPT-5.6 and 21.3% for Opus over
  the remaining 26,644, and 47.4% for Qwen. The rest are interfaces drawn without a usable tree, such as one container for a whole screen, links inside a text run -
  which no prompt can fix.
- **When the tree does name the element, the model is right about 85% of the time**, and the ~10% it gets wrong is the part no fallback recovers: see the cascade. 
  The phrasing barely moves the right-answer rate for GPT-5.6 and Opus (57–58% under all three); `intent` draws a few more wrong answers.
- **Qwen2.5-VL: close from the screenshot, far behind from the tree.** From the screenshot it is 6.7 points behind Opus (84.6% against 91.3%); 
  from the tree it is 21 points behind (35.8% against 57.2%), answers right a third of the time, and moves with the wording (40% under `name`, 33% under `label`). 
  4,344 of its XPaths (13.7% of requests) compile but select nothing, fourteen times GPT-5.6's 304:
  they name attributes the tree does not have. On 6 further requests, left out of the table, it repeated one path step until the 2,048-token cap; 
  GPT-5.6 and Opus produced no such answer in 31,704 (see [Answers stuck in a repetition loop on the tree track](#answers-stuck-in-a-repetition-loop-on-the-tree-track)).

Every row's `xml_outcome` and the per-file counts (`summary.xml_outcomes`) are in the result files;
[On the accessibility tree](#on-the-accessibility-tree) explains what each outcome means and how the prompt is handled.

### Tree first, screenshot as the fallback

A locator in the field would try the tree first and show the model the screenshot only when the tree yields nothing.
No new request is needed to measure that: the two tracks above are joined per (element, phrasing) by `benchmark/score_cascade.py`.
Every request falls into exactly one of lines 1, 2, 4, 5 or 6; lines 3, 7 and 8 are sums. 
Over all three phrasings, 31,704 requests per model (31,698 for Qwen, whose 6 tree errors are left out), percentages of that total:

| # |                                              |            GPT-5.6 |    Claude Opus 4.8 |      Qwen2.5-VL-7B |
|---|----------------------------------------------|--------------------|--------------------|--------------------|
| 1 | tree answered correctly                      |     58.1% (18,433) |     57.2% (18,125) |     35.8% (11,339) |
| 2 | tree answered incorrectly                    |      10.5% (3,327) |       9.5% (2,999) |       9.8% (3,098) |
| 3 | tree did not answer → sent to the screenshot |      31.4% (9,944) |     33.4% (10,580) |     54.4% (17,261) |
| 4 | screenshot answered correctly                |      30.5% (9,675) |      30.2% (9,575) |     45.9% (14,555) |
| 5 | screenshot answered incorrectly              |         0.9% (269) |       3.2% (1,005) |       8.5% (2,702) |
| 6 | screenshot did not answer                    |             0% (0) |             0% (0) |           0.0% (4) |
| 7 | **correct overall** (1 + 4)                  | **88.7% (28,108)** | **87.4% (27,700)** | **81.7% (25,894)** |
| 8 | **incorrect overall** (2 + 5 + 6)            |      11.3% (3,596) |      12.6% (4,004) |      18.3% (5,804) |

By phrasing, line 7 is 89.2% / 90.5% / 86.3% for GPT-5.6, 86.8% / 89.4% / 85.9% for Opus and 81.5% / 87.4% / 76.2% for Qwen
(`name` / `label` / `intent`): the same ordering as on the screenshot. All eight lines per phrasing are in the details block at the end of this section.

How to read it:

- **A wrong XPath is final.** The fallback fires on *not answered*, never on *answered wrongly*, so line 2 is lost for good and
  the cascade can score below the screenshot alone. It does here for every model: GPT-5.6's screenshot answers alone are right
  on 96.1% of these same 31,704 requests, the cascade on 88.7%; Opus 92.0% against 87.4%; Qwen 86.0% against 81.7%. What line 2
  loses (10.5% / 9.5% / 9.8%) outweighs what the screenshot gets wrong in line 5 (0.9% / 3.2% / 8.5%).
- **The screenshot answers come from each model's full vision run**, graded by the same centroid rule as the screenshot
  table above (`pixels` prompt); the cascade uses them only on the rows the tree did not answer. On those rows the screenshot 
  is right 97.3% of the time for GPT-5.6, 90.5% for Opus and 84.3% for Qwen - the rows the tree could not name are not hard from the screenshot.
- **Line 6 is 0 by construction, and that is the right reading for this dataset.** Every element the cascade asks about is
  on the screen, so it is in the human ground truth; "not found" is never the correct answer here, and letting the model
  decline could only turn a request in line 4 or 5 into a miss. The screenshot prompt therefore allows no "not found"
  answer, and line 6 holds only parse failures and transport errors: none for GPT-5.6 and Opus, and for Qwen the 4 requests it
  answered in prose with no box. What a model does when the element is genuinely absent is a different question, 
- and needs its own set of absent-element queries to measure.
- **The keyboard sits in line 3.** The ~4,850 keyboard requests GPT-5.6 and Opus could not answer from the tree (4,643 for Qwen)
  all go to the screenshot, where they are handled like any other control. The cascade therefore recovers the data gap, and
  lines 4–5 should be read with that in mind.

The sublines of 3 and 6 and the JSON behind the table are in [`reference-results/`](reference-results/) and described in
[`benchmark/README.md`](benchmark/README.md#score_cascadepy--tree-first-screenshot-as-the-fallback).

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

### When it is wrong, what did it do?

A failed answer is not one thing. Judged from the centre of the returned box against every human-drawn element on the same screen, 
over the same 10,568 elements and the `name` phrasing as the screenshot table:

|                                                                  | GPT-5.6 | Claude Opus 4.8 | Qwen2.5-VL-7B | GUI-Owl-1.5-8B |
|------------------------------------------------------------------|---------|-----------------|---------------|----------------|
| right element (centre inside it)                                 | 96.2%   | 91.3%           | 84.6%         | 81.8%          |
| **wrong element** — centre inside a *different* labelled element | 1.2%    | 3.2%            | 8.5%          | 7.7%           |
| near miss — overlaps the right element, centre just outside      | 1.8%    | 4.6%            | 4.6%          | 3.2%           |
| empty space — centre on no labelled element                      | 0.8%    | 1.0%            | 2.3%          | 7.4%           |
| declined — answered in prose, no box                             | 0       | 0               | 2 answers     | 0              |

 

When Qwen is wrong it has mostly chosen another control; 
Opus is wrong less often and, when it is, as likely to have drawn a loose box around the right one. 
GUI-Owl returned a bare click point for 8,840 of its 10,568 answers and a box for 1,728; when it misses, 
it lands on empty space almost as often as on another control — a point has no extent to overlap the right element with, 
so a near miss for it is rarer by construction. 

This table is about elements that are on the screen, which is the locator's job: given a name, find it. The benchmark asks for a 
box and nothing else, so *declined* counts a model breaking format rather than a considered "not there" — with this prompt, a model that 
cannot find the element still has to guess, and the guess lands in one of the rows above. The complementary question, whether a model 
says "not found" when the element is genuinely absent, is a separate measurement with its own set of absent-element queries; the 
annotator's [92 flagged elements](reference-results/flagged-by-annotator.md) are its starting point. The tree track is the one place
this benchmark does let a model decline; `NOT_FOUND` is an invited answer there, which is why it's *not answered* line is the
model's own call rather than a format slip.

Every result file carries the class per row (`answer_class`, with the id of the element chosen instead) and the counts (`summary.answer_classes`), and the 
[viewer](https://dataset-review-ui-test.kobiton.com/compare) filters on them — pick *Wrong element* and each screenshot shows the 
asked-for box and the chosen one side by side. The worst cases of each kind, with those links, are listed per model under 
[`reference-results/failures/`](reference-results/failures/) — what was asked, what was chosen instead — generated by `benchmark/list_failures.py`.

### Answers stuck in a repetition loop on the tree track

A tree-track answer is one XPath; the longest GPT-5.6 or Opus returned in 31,704 requests is 421 characters.
Qwen2.5-VL-7B produced 24 answers (0.08%) that did not terminate: a path step such as `/android.widget.FrameLayout[@resource-id='…']` repeated up to 60 times, 
or a `parent::` chain 70 deep, until the 2,048-token cap.

At the endpoint's 20–25 tokens per second one such answer takes 65–115 seconds, which is how the run's first pass came to record 10 client timeouts; 
a resume re-asked them and 6 hit the cap again, the run's 6 `error` rows. 
The 24 fall on 11 screens whose trees are larger than the median but inside the normal range; 
on the same 24 requests GPT-5.6 and Opus each answered 14 correctly, with XPaths of at most 178 characters. 
No figure moves — the answers are already *not answered* or excluded — but the 24 took 9.1% of the run's output tokens and held other requests in the queue for over a minute. 
The rows, the full answers and the arithmetic are in [`reference-results/repetition-loops/`](reference-results/repetition-loops/).

### How much the wording matters

Every element carries three descriptions — `name`, `label`, `intent` — and every model was asked all three. 
The screenshot table shows `name`; this section is what the other two add.

All four columns are on the human ground truth, over the same 10,568 elements as the screenshot table.

**Note:** GUI-Owl's centroid and its IoU are answers to two different questions. 
It is an agentic model trained to emit a click point `(x, y)`, so 84.6% of its answers carry no box at all, and only 17.9% of the boxes it does return reach IoU ≥ 0.5. 
A point has no area, so it scores zero on IoU by construction. 
If what you need is a bounding box, coverage is a **gate**, not a metric, and those two are not on the same leaderboard:
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

All four find `label` easiest, it is the phrasing that repeats the element's visible text most often. 
The hardest phrasing splits two and two: GPT-5.6 and Qwen on `intent`, Opus and GUI-Owl on `name` - 
Opus by a hair (91.26% against 91.29%), GUI-Owl by three points.

On the tree the wording hardly matters to the right-answer rate for GPT-5.6 and Opus - 57–58% under every phrasing - 
because what decides a tree request is whether the element has a node at all, not how it was described. 
Qwen is the exception: 40% / 33% / 35% under `name` / `label` / `intent`. 
What moves is the wrong-answer rate: `intent`, the phrasing with no text handle, draws 12.9% wrong from GPT-5.6 against 8.6% under `label`.

A single-phrasing headline flatters a model, because `name` is usually the element's visible text and `intent` deliberately 
carries no text handle at all. 
Scoring all three is what makes the figure describe the ways a tester might actually phrase a locator rather than the easiest one.

**And the average still flatters it.** 
Averaging the three counts an element as 2-of-3 correct; 
what a test suite needs is the element working *whatever* the tester typed. 
Measured per element, on the screenshot:

|                         | average centroid | passes under **all three** |
|-------------------------|------------------|----------------------------|
| GPT-5.6                 | 96.10%           | **93.05%**                 |
| Claude Opus 4.8         | 91.97%           | **87.07%**                 |
| Qwen2.5-VL-7B-Instruct  | 85.99%           | **75.58%**                 |
| GUI-Owl-1.5-8B-Instruct | 84.57%           | **72.30%**                 |

So the figure to plan reliability against is three points below the headline for the strongest model here, five for Opus, and
ten to twelve for the open-weight ones — the gap widens as the model weakens, which is the opposite of what an average suggests. 
(`summary.agreement` in every multi-phrasing result file carries these counts; on the tree it is 54.6%, 52.9% and 23.4%, in the table above.)

Full summaries for every run — token totals and every field described below — are in [`reference-results/`](reference-results/). 
Latency is recorded per run, but each figure describes its own serving setup — it is not comparable across rows, and not a production figure for any of them.

---

## Quickstart

Python 3.9+, and an OpenAI-compatible endpoint serving a model: a vision model for the screenshot track, any text model for the tree track.

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

The same command with `--input xml` scores the tree track instead — no image is sent, so any text model will do — and writes`xml-<model>-…` rather than `vision-<model>-…`:

```bash
python benchmark/run_vision_benchmark.py --input xml \
    --base-url http://localhost:8000 \
    --model qwen2.5-vl \
    --description-index 0 1 2
```

With one file of each kind for the same model, the cascade is a join and costs nothing. `--only-from` buys the screenshot
answers the cascade needs and no others — the rows the tree did not answer — for a model that has no full screenshot run:

```bash
python benchmark/run_vision_benchmark.py --base-url http://localhost:8000 --model qwen2.5-vl \
    --description-index 0 1 2 --only-from benchmark-results/xml-qwen2.5-vl-3phrasing-GT-v1-….json

python benchmark/score_cascade.py \
    benchmark-results/xml-qwen2.5-vl-3phrasing-GT-v1-….json \
    benchmark-results/vision-qwen2.5-vl-3phrasing-ONLY-FROM-xml-GT-v1-….json
```

Output lands in `benchmark-results/<vision|xml>-<model>-GT-<dataset>-<timestamp>.json`. 
Anything that makes a number mean something narrower — a `--limit` pilot, a stopped run, an endpoint that answered as a different model, 
a screenshot run restricted to the tree's leftovers — is marked in the filename too; [`benchmark/README.md`](benchmark/README.md) lists them.

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

**llama.cpp** — requests are issued grouped by screenshot, so llama.cpp's prompt cache encodes each image once and 
every further request about that screenshot costs a fraction of a second; with one slot this is what makes a 
full run take hours rather than days.

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

> **Paid endpoints.** `--rpm` caps request starts per minute across all workers and `--max-requests`
> caps one invocation's model calls, so a full run can stay under an account's per-minute limit and be
> spread overnights under its spend limit; a capped run resumes with the same command. On the tree
> track the shared prefix - the instructions and the tree are sent so both OpenAI's and Anthropic's
> prompt caches pick it up, and cache writes and reads are recorded per run.

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

|                     | `dataset-v1.jsonl`                          | `dataset-v1-gpt-5.6.jsonl` | `dataset-v1-human.jsonl` |
|---------------------|---------------------------------------------|----------------------------|--------------------------|
| Labelled by         | Claude Opus 4.8                             | GPT-5.6                    | a human annotator        |
| Elements            | 10,314                                      | 10,647                     | 11,914                   |
| With a bounding box | 10,307 (99.93%)                             | 10,647 (100%)              | 11,914 (100%)            |
| Screenshots covered | 841                                         | 839                        | 841                      |
| Descriptions        | 3 per element (`name` / `label` / `intent`) | same                       | same                     |

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

A fourth file, `dataset-v1-human-gpt.jsonl`, is the human file's 10,568 `gpt-` elements on their own.
The population every figure in [Results](#results) is over (`summary.by_source.gpt` of a run on the full file). 
The tree-track and cascade reference results were run on it directly, so their filenames carry `-GT-v1-human-gpt-`.

**Beside each screenshot sits its page source**: `data/xml/<screenshot_id>.xml`, the UiAutomator dump captured at the same moment, 841 of them. 
It is what the tree track sends to the model and runs the returned XPath on. Two things about it shape every tree figure:

- It covers the **application window only**. A tree read on a live device also holds the on-screen keyboard; here a key on the
  keyboard has no node, so the ~1,700 elements that are keys (5,060 of the 31,704 requests) cannot be answered from the tree
  by any model. The scorers count them in a footnote rather than removing them.
- It is the tree **as captured**, not as filtered. The model sees a filtered copy; the XPath it returns is run on the full dump,
  as a driver would run it on the live tree, which is also where a positional index can land on a different node than the one the model counted.

Screenshots are 841 Android, 1080×2400 throughout, crawled from **public app-store packages**. 
Every element was found, boxed and described by a frontier model, then spot-checked before the dataset was accepted.

**Ground truth from a model is a stated limitation, not a hidden one.** 
It buys ~10,000 elements instead of the few hundred hand-labelling would have produced, and the box convention is at least *consistent*, 
which is what makes the centroid/IoU split legible. 
The human labelling is what turned that limitation from assumed into measured — 
the label-quality numbers and what the adjudication found are the headline of [Results](#results).

Everything is in the repository: all four `.jsonl` files, their stats sidecars, all 841 PNGs under `data/images/`, named
`<screenshot_id>.png`, and the 841 dumps under `data/xml/`. 
A clone is ~320 MB and is everything the benchmark needs; no separate download step required.

---

## What is being measured

Two inputs, one scoring rule. Whether the answer is a box the model drew on the screenshot or the bounds of the node its
XPath selected, it is graded against the human's box the same way, and two metrics are always computed. `--metric` only
picks which one is the headline:

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

### On the screenshot

The model is shown the screenshot and one description and asked for a bounding box, nothing else.
**The prompt deliberately states no box convention at all.** 
Loading the labeler's rules into it would coach models toward the labeler's answer and measure agreement with our annotator 
rather than element localization. Three things about this input decide whether a number can be trusted, and each one cost us a run to learn.

#### Pixels, or a 0–1000 grid?

**Which coordinate scale a model answers on is a property of the model, and it cannot be recovered from a single answer.** 
On a 1080×2400 screenshot `[67, 91]` is a legal pixel pair *and* a legal point on a 0–1000 grid, 
and the two readings are 2.4× apart on the y-axis. 
Qwen2.5-VL answers in the screenshot's own pixels; GUI-Owl answers on a 0–1000 grid.
Reading the grid as pixels gave us this, on a run that looked perfectly clean — 
30,921 rows, 4 errors, every answer short and well-formed:

| GUI-Owl, same 30,921 answers | centroid   | IoU ≥ 0.5 |
|------------------------------|------------|-----------|
| read as pixels               | **10.56%** | 0.95%     |
| read on the 0–1000 grid      | **87.80%** | 2.93%     |

So: **pilot 200 elements and read `summary.scale_check` before trusting anything.** 
If it flags, find the model's convention from its model card, add it to `COORD_GRIDS` in `benchmark/scoring/coords.py`, 
and re-score the pilot with `--rescore` rather than paying for the run twice. 
`scale_check` flags and never corrects, and `suspect: false` means "no evidence here" rather than "correct" — 
[`benchmark/README.md`](benchmark/README.md) explains why, and how the check's sensitivity depends on aspect ratio.

#### How you ask for the coordinates changes the score

The other half of the same problem: the grid above is how a model's answer is *read*, and this is how it was *asked*. 
`--prompt-style pixels`, the default, states the image size and asks for integer pixels. 
`--prompt-style normalized` asks for floats in [0, 1] and states no size.

That sounds cosmetic and is not. 
A model that ignores the instruction is unaffected — Qwen2.5-VL answers in pixels either way and the parser rescales — 
but a model that *obeys* it can be ruined by it. 
Over 200 elements × 3 phrasings, changing nothing else, one frontier model went from **7.79%** centroid to **90.33%**:

| same 200 elements, same ground truth   | centroid   | IoU ≥ 0.5  | median `dy` |
|----------------------------------------|------------|------------|-------------|
| `normalized` — floats, no image size   | 7.79%      | 2.27%      | +0.0816     |
| `pixels` — integers, image size stated | **90.33%** | **61.67%** | +0.0004     |

Both runs had zero errors and returned a box on 100% of requests. The model was answering every time; 
under one prompt it was answering in a frame it had never been given — every prediction landing ~8% of the screen height too low. 
That is a harness result, not a model result, and it is the single most expensive thing to get wrong here.

So `prompt_style` is recorded in every result file, and two runs either side of it are not comparable. 
**The two open-weight rows above were measured under `normalized`** and are not understated by it, 
because Qwen ignores the instruction — but GUI-Owl has not been measured under `pixels` at all. 
[`benchmark/README.md`](benchmark/README.md) has the 10-element control that splits the cause into its two halves.

#### Screenshots go out at full resolution

`--max-image-dim` defaults to `0`, and downscaling is not a benchmark setting: a comparison is meaningful only when 
every model saw the screenshot a consumer would actually send. 
We learned it expensively — an early run shrank every screenshot to a 1080px long edge and reported **70.07%** centroid for Qwen2.5-VL 
where native resolution reports **84.40%** on the same phrasing, same ground truth, same metric. 
A small control is ~34px across natively and ~15px after that resize, i.e. below what the model can resolve.

Treat a non-zero `max_image_dim` in a result file the way you would treat `PARTIAL` — check what the number describes before quoting it. 
Native costs roughly 4–5× the image tokens, and the image is essentially the whole input; `benchmark/README.md` has the measurements.

### On the accessibility tree

The page source is filtered - leaf nodes with no `text`, `content-desc`, `resource-id` or `hint` go, over-deep branches go,
every attribute outside identifiers, text, state and bounds goes, invisible nodes stay and sent as text with the description.
The model answers **one XPath or `NOT_FOUND`**. The XPath is run on the *full* dump, as a driver would run it on the live tree,
the first node it selects is taken, as a driver's find-element takes it, and that node's `bounds` become the predicted box.
From there the centroid and IoU rules above apply unchanged.

What comes back is one of four things, recorded per row as `xml_outcome`, and **none of them is an error**. They are what the
model answered, and a resume never pays to ask again:

| `xml_outcome`   | meaning                                                   | in the tables above          |
|-----------------|-----------------------------------------------------------|------------------------------|
| `answered`      | the XPath selected at least one node; graded on the first | right or wrong, by geometry  |
| `not_found`     | the model said `NOT_FOUND` — the one outcome it *decided* | not answered                 |
| `invalid_xpath` | the text does not compile as XPath 1.0                    | not answered                 |
| `no_match`      | valid XPath, zero nodes on this tree                      | not answered                 |

Two consequences for reading a tree figure. First, *not answered* mixes a decision (`NOT_FOUND`) with two kinds of failure a
driver would also reject, which is why the tables split them. Second, in a tree-then-screenshot cascade every outcome but
`answered` falls through to the screenshot, and `answered` is final whether it was right or wrong — the fallback cannot see
that a selected node is the wrong one.

**The instructions are a file, not code.** The model reads `benchmark/model/prompts/xpath-android.txt` before the tree;
`--xml-prompt` points a run at another file, and the harness appends the filtered tree and the description itself, in that
order - tree first, so the ~38 requests about one screenshot share a cacheable prefix. Every result records the file's name
and a hash of its text, and a checkpoint refuses to resume under a different prompt, so two tree results can always be told
apart by what they were asked with. **The default file here is not the prompt our reference figures were measured with.** 
That one ships inside a product and stays out of this repository; the default has the same shape, including the role, the 
strict output contract, the grounding rules, the Android attribute priority, the ordinal pattern - in fewer words,
so a run with it will differ from the published `xml-…` numbers by the prompt rather than by the harness.

Request by request, with the measurements behind the caching and the prompt order, in [`benchmark/README.md`](benchmark/README.md#--input-xml--the-xml-track).

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
`summary.by_source` has the figures split by which labelling the element came from (`gpt-`/`opus-` ids — the screenshot table above is `by_source.gpt`), 
`summary.answer_classes` and each row's `answer_class` say what the model did when it was wrong (see [When it is wrong](#when-it-is-wrong-what-did-it-do)), 
and every row keeps the model's own untruncated answer in `raw` plus the `img_w`/`img_h` it was normalised against. 
A tree-track file (`xml-…`) adds `summary.xml_outcomes`, each row's `xpath` and `xml_outcome`, and the prompt's name and hash;
`score_cascade.py --json` writes the eight cascade lines with their sublines, overall and per phrasing. 
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
  Re-scoring the same `raw` proves which one moved. On a tree-track file it re-runs every XPath
  against the dumps instead of reparsing coordinates.
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

| Step                   | Calls             |                                             |
|------------------------|-------------------|---------------------------------------------|
| 1 · extract elements   | 1 per screenshot  | identify every interactive element          |
| 2 · detect boxes       | **1 per element** | dominates the cost                          |
| 3 · write descriptions | 1 per screenshot  | three phrasings each                        |
| 4 · assemble           | 0                 | merge into one JSONL, write a stats sidecar |

Everything it produces stays on your machine, under `data/`:

```
data/my-dataset.jsonl          the labels, in the same shape as the shipped dataset
data/my-dataset-stats.json     counts, plus which provider and model did the labelling
data/checkpoints/…             per-screenshot progress; deleted once the run completes cleanly
```

Those paths are anchored on the repository root, not on the directory you run the command from,
so a resume finds the earlier run's progress wherever you start it. Point the benchmark at the
result with `--dataset data/my-dataset.jsonl` and `--images-dir path/to/screenshots`. For the
tree track you also need each screenshot's page source as `<screenshot_id>.xml` — a UiAutomator
dump taken at the same moment — in the directory `--xml-dir` names.

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
  run_vision_benchmark.py   the benchmark, both tracks; --help lists every flag
  score_cascade.py          tree-then-screenshot cascade from an xml-… and a vision-… result
  README.md                 every metric, every field, and why each exists
  cli/ engine/ artifacts/   the flags · the run · checkpoint, result JSON, re-scoring, cascade rules
  model/ scoring/ workload/ the system under test · what an answer means · what gets scored
  model/prompts/            the tree track's instructions file (see --xml-prompt)
pipeline/                   the four labelling steps
run_pipeline.py             pipeline entrypoint
data/
  dataset-v1.jsonl          ground truth, labelled by Opus          ← committed
  dataset-v1-gpt-5.6.jsonl  the same screens, by GPT-5.6            ← committed
  dataset-v1-human.jsonl    a human adjudicating the two            ← committed
  dataset-v1-human-gpt.jsonl  its 10,568 gpt- elements on their own ← committed
  images/                   841 screenshots, <id>.png               ← committed
  xml/                      841 page-source dumps, <id>.xml         ← committed
reference-results/          our published summaries: vision-…, xml-…, cascade-…  ← committed
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
