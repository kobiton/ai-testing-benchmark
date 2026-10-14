# Run configurations

One JSON file per model and track. These are the configurations the files under [`results/`](../results/) were made with, and the way to run the benchmark:

```bash
ai-testing-benchmark run --config config/vision/gpt-5-6-terra.json           # the screenshot track
ai-testing-benchmark run --config config/xml/gpt-5-6-terra.json              # the tree track
ai-testing-benchmark cascade --config config/cascade/gpt-5-6-terra.json      # joins the two results
ai-testing-benchmark run --config config/vision/qwen2-5-vl-7b.json --limit 200   # a pilot: flags override the file
```

```
config/
  vision/   claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b  gui-owl-1-5-8b
  xml/      claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b
  cascade/  claude-opus-4-8  gpt-5-6-terra  qwen2-5-vl-7b
```

Three rules, enforced by the loader (`src/ai_testing_benchmark/config.py`):

- **Every key is checked before anything is loaded.** An unknown key, a missing required key or a
  value of the wrong type stops the run with the key named.
- **No credential in the file.** `endpoint.api_key_env` names the environment variable the key is
  read from (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); the run refuses to start when it is unset,
  except under `--dry-run` and in `cascade`, which call no model. The
  self-hosted configs name no variable: a local server started without a key wants no auth header.
- **The result records what was used.** Every result file carries a `config` block: the file's
  name and hash, and the resolved values after flag overrides and defaults.

Relative paths resolve against the project root (the checkout holding `pyproject.toml`), not the
working directory.

## A run config

```json
{
  "model": "gpt-5.6-terra",
  "input": "screenshot",
  "endpoint": {"base_url": "https://api.openai.com", "api_flavor": "openai", "api_key_env": "OPENAI_API_KEY"},
  "dataset": {"path": "data/labels/ground-truth.jsonl", "annotation_origin": ["gpt-5.6"],
              "images_dir": "data/images"},
  "description_indices": [0, 1, 2],
  "prompt": {"style": "pixels"},
  "coordinates": {"format": "corner", "grid": -1},
  "inference": {"max_tokens": 2048, "temperature": 0, "thinking_budget": -1, "max_image_dim": 0,
                "timeout": 120, "workers": 5, "rpm": 60},
  "metric": "centroid",
  "output_dir": "benchmark-results"
}
```

| key                         | required         | flag                  | meaning                                                                                                                                                                  |
|-----------------------------|------------------|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`                     | yes              | `--model`             | the name to ask the endpoint for; the result records who actually answered as `served_model`                                                                             |
| `input`                     | yes              | `--input`             | `screenshot` (the vision track) or `xml` (the tree track)                                                                                                                |
| `endpoint.base_url`         | yes              | `--base-url`          | an OpenAI-compatible server; `/v1/chat/completions` is appended (`/v1/messages` under `anthropic`)                                                                       |
| `endpoint.api_flavor`       | yes              | `--api-flavor`        | `proxy` (vLLM, llama.cpp, Ollama, gateways), `openai` (api.openai.com) or `anthropic`. Never inferred from the URL                                                       |
| `endpoint.api_key_env`      | no               | `--api-key`           | the environment variable holding the key. Omit for a local server without one                                                                                            |
| `dataset.path`              | yes              | `--dataset`           | the ground truth, one element per line                                                                                                                                   |
| `dataset.annotation_origin` | no               | `--annotation-origin` | keep only elements whose `annotation.origin` is listed. The published figures use `["gpt-5.6"]`, the 10,568 elements every model has answers for. Omit for every element |
| `dataset.images_dir`        | no               | `--images-dir`        | `<images_dir>/<screenshot_id>.png` (default `data/images`)                                                                                                               |
| `dataset.xml_dir`           | no               | `--xml-dir`           | tree track: `<xml_dir>/<screenshot_id>.xml` (default `data/xml`)                                                                                                         |
| `dataset.limit`             | no               | `--limit`             | score at most N elements, spread evenly across the dataset; 0 = all                                                                                                      |
| `description_indices`       | yes              | `--description-index` | which phrasings to ask: 0 = `name`, 1 = `label`, 2 = `intent`                                                                                                            |
| `prompt.style`              | screenshot track | `--prompt-style`      | `pixels` (states the image size, asks for integers) or `normalized` (floats in [0, 1])                                                                                   |
| `prompt.xml_prompt`         | no, tree track   | `--xml-prompt`        | the instructions file sent before the tree; omit for the packaged default                                                                                                |
| `coordinates.format`        | no               | `--coord-format`      | how `(x, y)` in an answer is read: `corner` (top-left) or `center`                                                                                                       |
| `coordinates.grid`          | no               | `--coord-grid`        | divide pixel-scale answers by N on both axes; `-1` picks it per model (GUI-Owl answers on 0–1000), `0` forces the screenshot's own pixels                                |
| `inference.max_tokens`      | yes              | `--max-tokens`        | output cap; a truncated answer is an error, not a box                                                                                                                    |
| `inference.temperature`     | yes              | `--temperature`       | `0` is greedy; `-1` sends none, for a model that rejects the parameter                                                                                                   |
| `inference.thinking_budget` | no               | `--thinking-budget`   | `budget_tokens` for chain-of-thought models; `-1` leaves the model's default                                                                                             |
| `inference.max_image_dim`   | no               | `--max-image-dim`     | `0` sends the screenshot at full size, which is the benchmark setting                                                                                                    |
| `inference.timeout`         | no               | `--timeout`           | per-request seconds                                                                                                                                                      |
| `inference.workers`         | yes              | `--workers`           | concurrent requests; every latency figure is measured at this value                                                                                                      |
| `inference.rpm`             | no               | `--rpm`               | cap request starts per minute across all workers, for a paid endpoint                                                                                                    |
| `inference.max_requests`    | no               | `--max-requests`      | model calls per invocation, then a `PARTIAL` result; the same command resumes                                                                                            |
| `metric`                    | no               | `--metric`            | the headline number: `centroid` (default) or `iou`. Both are always computed                                                                                             |
| `output_dir`                | no               | `--output-dir`        | where result files and checkpoints go                                                                                                                                    |

`prompt.style` is refused on the tree track and `prompt.xml_prompt` on the screenshot track: each track has exactly one way of asking.

## A cascade config

```json
{
  "model": "gpt-5.6-terra",
  "xml_config": "config/xml/gpt-5-6-terra.json",
  "vision_config": "config/vision/gpt-5-6-terra.json",
  "annotation_origin": ["gpt-5.6"],
  "with_vision_alone": true,
  "results_dir": "benchmark-results"
}
```

| key                           | meaning                                                                                                                                                                                 |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`                       | the model both halves come from                                                                                                                                                         |
| `xml_config`, `vision_config` | the two run configs; the newest complete `xml-<model>-…` / `vision-<model>-…` result of each in `results_dir` is joined. Pilots, stopped runs, subsets and `-SERVED-` files are skipped |
| `xml_result`, `vision_result` | instead of a run config: a result file outright                                                                                                                                         |
| `annotation_origin`           | which elements to keep, read from the run config's dataset; defaults to the xml run's own selection                                                                                     |
| `with_vision_alone`           | also report the screenshot answers scored on every row, for comparison with the cascade                                                                                                 |
| `results_dir`                 | where the two results are looked for (default `benchmark-results`)                                                                                                                      |

The join needs both files' per-request rows. The files under `results/` are summaries, 
so a cascade is recomputed from your own runs, not from them.

## What a config does not decide

- **The tree prompt the published `xml` results were measured with is not in this repository.**
  It is a production locator's template; the packaged default has the same shape in fewer words,
  so a run with these configs will differ from `results/xml/*` by the prompt. Every result records
  the prompt's name and hash (`xml_prompt`, `xml_prompt_sha256`), so two tree results can always
  be told apart.
- **The endpoint behind a self-hosted config is yours.** `qwen2-5-vl-7b` and `gui-owl-1-5-8b`
  point at `localhost`; the published figures came from a vLLM deployment on a GPU cluster and
  from llama.cpp on a Mac mini respectively, and the result files say so.
