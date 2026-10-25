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

It contains **841 real Android screenshots**, each with its page source, and **11,914 human-verified elements**
labelled by two frontier models independently and adjudicated by a human annotator, each described in three different ways;
the **10,568** every model has an answer for are the benchmark population.
The runner works with any **OpenAI-compatible endpoint**, including vLLM, llama.cpp, Ollama, and hosted APIs,
and every run is described by a JSON configuration that the result file records.

It exists because we needed to choose a self-hosted model and a locator strategy, and wanted a benchmark tailored to the question we actually needed to answer.
Published so you can check our numbers and score your own model on the same corpus.

---

## Results

**Scored against the human-adjudicated ground truth** ([`data/labels/ground-truth.jsonl`](data/labels/ground-truth.jsonl)):
an independent human annotator settled every element the two frontier labellings disagreed on and verified the ones they agreed on
([how it was built](docs/annotations.md)).
Every figure below is over the same **10,568 elements** — the ones with `annotation.origin = gpt-5.6`, which every model here has an answer for —
under the `name` phrasing unless a table says otherwise.
Correct means that the center of the returned box, or of the node selected by the XPath, falls within the human-annotated box for that element.

### At a glance

Three ways to locate the same element, three models that were measured under all three.

|                        | Screenshot alone | Tree alone | Tree first, screenshot as the fallback |
|------------------------|------------------|------------|----------------------------------------|
| **GPT-5.6**            | 96.2%            | 58.1%      | 89.2%                                  |
| **Claude Opus 4.8**    | 91.3%            | 57.2%      | 86.8%                                  |
| **Qwen2.5-VL-7B**      | 84.6%            | 35.8%      | 81.5%                                  |

Tree alone and the cascade are not the same measurement as the screenshot alone, and the gap between the columns is the point:
a third of the elements have no node the tree can name (the keyboard, and interfaces drawn without a tree),
and a wrong XPath is final, the fallback never fires on it.

### Centroid Accuracy Explanation

**A prediction passes the centroid check when the center of its bounding box falls inside the human-annotated target box.** When automation uses that center as its tap point, this checks whether the tap would land inside the intended control.

![Examples of centroid correctness: predicted centers inside the target pass; centers outside the target fail.](https://github.com/user-attachments/assets/3224848f-cb3a-474b-82b0-d07187e1b099)

In the visual, the **teal box** marks the human-annotated target, the **yellow box** shows the predicted bounds, and the **dot** marks the predicted center.

On the left, the green dots fall inside the target, so both predictions pass. The boxes do not need to line up perfectly. On the right, the red dots fall outside the target, so both predictions fail—even when the boxes overlap.

**Centroid accuracy is the percentage of evaluated requests that pass this check.** The same rule applies to screenshot predictions and the bounds of elements selected from the UI tree, giving the screenshot, tree, and cascade approaches a common scoring criterion.

This metric checks where the proposed tap would land. A live test must still confirm that tapping the control produces the expected result.

### From the screenshot

|                             | tap-point correct (centroid) | box correct (IoU ≥ 0.5) | median IoU | returned a box | asked for |
|-----------------------------|------------------------------|-------------------------|------------|----------------|-----------|
| **GPT-5.6**                 | **96.2%**                    | **81.7%**               | 0.890      | 100%           | pixels    |
| **Claude Opus 4.8**         | 91.3%                        | 61.2%                   | 0.617      | 100%           | pixels    |
| **Qwen2.5-VL-7B-Instruct**  | 84.6%                        | 55.4%                   | 0.553      | 99.98%         | pixels    |
| **GUI-Owl-1.5-8B-Instruct** | 81.8%                        | 2.1%                    | 0.216      | 16.4%          | floats    |

Every row is a benchmark answer: screenshot + one description in, box out. All three phrasings were asked and `name` is shown;
averaged over the three, GPT-5.6 scores 96.10% with 93.05% of elements passing under all three, Opus 91.97% and 87.07%,
Qwen 85.99% and 75.58%. GUI-Owl returns a bare click point on 84% of requests, so its IoU is over the rest; centroid is its figure to read.

### From the accessibility tree

The same elements and phrasings, no image. Over all three phrasings — 31,704 requests per model:

|                                              |        GPT-5.6 | Claude Opus 4.8 |   Qwen2.5-VL-7B |
|----------------------------------------------|----------------|-----------------|-----------------|
| answered, right element                      | 58.1% (18,433) |  57.2% (18,125) |  35.8% (11,339) |
| answered, wrong element                      |  10.5% (3,327) |    9.5% (2,999) |    9.8% (3,098) |
| not answered                                 |  31.4% (9,944) |  33.4% (10,580) |  54.4% (17,261) |
| ↳ of which `NOT_FOUND`                       |          9,637 |          10,251 |          12,757 |
| ↳ of which the XPath did not compile         |              3 |              46 |             160 |
| ↳ of which the XPath matched no node         |            304 |             283 |           4,344 |

5,060 of the 31,704 requests describe a key on the on-screen keyboard, which the page source does not contain, so no model can answer them from the tree.
Two models seven points apart on the screenshot are within a point of each other here; Qwen2.5-VL is close from the screenshot and far behind from the tree,
with fourteen times GPT-5.6's XPaths that compile but select no node.

### Tree first, screenshot as the fallback

The two tracks joined per (element, phrasing); every request falls into exactly one of lines 1, 2, 4, 5 or 6:

| # |                                              | GPT-5.6            | Claude Opus 4.8    | Qwen2.5-VL-7B      |
|---|----------------------------------------------|--------------------|--------------------|--------------------|
| 1 | tree answered correctly                      | 58.1% (18,433)     | 57.2% (18,125)     | 35.8% (11,339)     |
| 2 | tree answered incorrectly                    | 10.5% (3,327)      | 9.5% (2,999)       | 9.8% (3,098)       |
| 3 | tree did not answer → sent to the screenshot | 31.4% (9,944)      | 33.4% (10,580)     | 54.4% (17,261)     |
| 4 | screenshot answered correctly                | 30.5% (9,675)      | 30.2% (9,575)      | 45.9% (14,555)     |
| 5 | screenshot answered incorrectly              | 0.9% (269)         | 3.2% (1,005)       | 8.5% (2,702)       |
| 6 | screenshot did not answer                    | 0% (0)             | 0% (0)             | 0.0% (4)           |
| 7 | **correct overall** (1 + 4)                  | **88.7% (28,108)** | **87.4% (27,700)** | **81.7% (25,894)** |
| 8 | **incorrect overall** (2 + 5 + 6)            | 11.3% (3,596)      | 12.6% (4,004)      | 18.3% (5,804)      |

The cascade scores below the screenshot alone for every model (96.1% / 92.0% / 86.0% on the same requests), because a wrong XPath is final:
what line 2 loses outweighs what the screenshot gets wrong in line 5.

**The full tables** — per phrasing, the sublines, how the wording changes the score, what each result file holds — are in
[`results/README.md`](results/README.md); the summaries themselves are one file per model and track under [`results/`](results/).
What the models do when they are wrong, class by class and screenshot by screenshot, is in [`docs/error-analysis.md`](docs/error-analysis.md).
**[Browse every row in the viewer →](https://dataset-review-ui-test.kobiton.com/compare)**

---

## Quickstart

Python 3.9+, and an OpenAI-compatible endpoint serving a model: a vision model for the screenshot track, any text model for the tree track.

```bash
git clone https://github.com/kobiton/ai-testing-benchmark.git
cd ai-testing-benchmark

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Validate the dataset and write a result file — calls no model, costs nothing
ai-testing-benchmark run --config config/vision/qwen2-5-vl-7b.json --dry-run
```

A run is a JSON file under [`config/`](config/): the model, the endpoint, which rows of the dataset, the three phrasings,
how the model is asked and how its answers are read ([the schema](config/README.md)).
The shipped files are the configurations the published results were made with; copy one and change the model and endpoint
to score your own. The file never holds a key — `endpoint.api_key_env` names the variable (`cp .env.example .env` and fill it in).

Start with `--limit` — any flag overrides the file — the full dataset is 10,568 elements and three phrasings make that 31,704 requests:

```bash
ai-testing-benchmark run --config config/vision/qwen2-5-vl-7b.json --limit 200
```

`--limit` takes an even stride across the dataset rather than the first N rows, so 200 elements land on ~199 distinct screenshots instead of the first 19.
It is deterministic, so two pilots are comparable, and a later full run **reuses** the pilot's scores instead of paying for them again.
When the pilot looks sane, drop `--limit`; the tree track is the matching file under `config/xml/`, and with one result of each kind
the cascade is a join and costs nothing:

```bash
ai-testing-benchmark run --config config/vision/qwen2-5-vl-7b.json
ai-testing-benchmark run --config config/xml/qwen2-5-vl-7b.json
ai-testing-benchmark cascade --config config/cascade/qwen2-5-vl-7b.json
```

Output lands in `benchmark-results/<vision|xml>-<model>-…-GT-ground-truth-gpt-5.6-<timestamp>.json`, with the configuration recorded inside.
Anything that makes a number mean something narrower — a `--limit` pilot, a stopped run, an endpoint that answered as a different model —
is marked in the filename too. Runs resume from a checkpoint when re-run with the same command, and Ctrl+C keeps the score written so far.

**Endpoints.** Any server exposing `POST /v1/chat/completions` with image content works with `"api_flavor": "proxy"`:
vLLM (`vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 32768`, port 8000), Ollama (port 11434), llama.cpp (port 8080; requests are
grouped by screenshot so its prompt cache encodes each image once). A hosted API needs `"api_flavor": "openai"` or `"anthropic"`, which is never
inferred from the URL — the same base URL can front either shape, and guessing wrong fails per element in a way that reads like a model problem.
`inference.rpm` and `--max-requests` keep a paid run under an account's limits. Every flag, and the reasoning behind the defaults, is in
[`docs/methodology.md`](docs/methodology.md).

---

## Documentation

|                                                    |                                                                                                                                                                        |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`docs/methodology.md`](docs/methodology.md)       | what is measured and how: the two tracks, the metrics, every flag and result field, interrupting and resuming, re-scoring, the cascade, labelling your own screenshots |
| [`docs/annotations.md`](docs/annotations.md)       | how the ground truth was built from two model labellings and a human adjudication, and what the adjudication found                                                     |
| [`docs/error-analysis.md`](docs/error-analysis.md) | what a wrong answer looks like, per model and per class; the tree-track answers that never terminated; the elements the annotator refused                              |
| [`data/README.md`](data/README.md)                 | the dataset: fields, counts, the page sources, how to select a population                                                                                              |
| [`config/README.md`](config/README.md)             | the run configuration schema                                                                                                                                           |
| [`results/README.md`](results/README.md)           | the full result tables and how to read a result file                                                                                                                   |

---

## Repository layout

```
src/ai_testing_benchmark/   the package: `ai-testing-benchmark run | cascade | failures | pipeline`
  benchmark/                the harness — cli/ engine/ artifacts/ model/ scoring/ workload/, the cascade scorer, the failure browser
  pipeline/                 the four labelling steps that produced the dataset
  config.py                 JSON run configurations: schema, validation, what the result records
config/                     one configuration per model and track; the ones the results were made with
data/                       labels/ground-truth.jsonl · images/<id>.png · xml/<id>.xml
results/                    the published summaries: vision/ xml/ cascade/, one file per model
docs/                       methodology · annotations · error-analysis
pyproject.toml              `pip install -e .`; `pip install -e ".[pipeline]"` adds the labelling pipeline's dependencies
```

Both tools write only inside the repository, and everything they write is gitignored: `benchmark-results/` (one JSON per run, plus
`checkpoints/` so a run resumes) and `data/<name>.jsonl` (a dataset you label yourself). Nothing is uploaded anywhere and nothing is
read from a network location: the corpus is in the clone (~320 MB), and the only address the tools contact is the model endpoint you give them.

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
