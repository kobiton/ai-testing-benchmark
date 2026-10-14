# Error analysis — when a model is wrong, what did it do?

The headline tables in the [README](../README.md) say how often each model is right. This page is
about the rest: what a wrong answer looks like, screenshot by screenshot, and the one failure
mode that is not a wrong answer at all.

```
docs/error-analysis/
  gpt-5-6-terra.md       the failures of each model on the screenshot track, class by class,
  claude-opus-4-8.md     worst first, each row linking the screenshot in the viewer with the
  qwen2-5-vl-7b.md       asked-for box, the model's box and the control it chose instead
  gui-owl-1-5-8b.md
  repetition-loops/      the 24 tree-track answers of Qwen2.5-VL-7B that never terminated
docs/annotations/
  flagged-by-annotator.md  the 92 elements the human annotator refused to box
```

## Screenshot track: the four kinds of wrong

A failed answer is not one thing. Judged from the center of the returned box against every
human-drawn element on the same screen, over the same 10,568 elements and the `name` phrasing
as the README's screenshot table:

|                                                                  | GPT-5.6 | Claude Opus 4.8 | Qwen2.5-VL-7B | GUI-Owl-1.5-8B |
|------------------------------------------------------------------|---------|-----------------|---------------|----------------|
| right element (centre inside it)                                 | 96.2%   | 91.3%           | 84.6%         | 81.8%          |
| **wrong element** — centre inside a *different* labelled element | 1.2%    | 3.2%            | 8.5%          | 7.7%           |
| near miss — overlaps the right element, centre just outside      | 1.8%    | 4.6%            | 4.6%          | 3.2%           |
| empty space — centre on no labelled element                      | 0.8%    | 1.0%            | 2.3%          | 7.4%           |
| declined — answered in prose, no box                             | 0       | 0               | 2 answers     | 0              |

When Qwen is wrong it has mostly chosen another control; Opus is wrong less often and, when it
is, as likely to have drawn a loose box around the right one. GUI-Owl returned a bare click point
for 8,840 of its 10,568 answers and a box for 1,728; when it misses, it lands on empty space
almost as often as on another control — a point has no extent to overlap the right element with,
so a near miss for it is rarer by construction.

This table is about elements that are on the screen, which is the locator's job: given a name,
find it. The benchmark asks for a box and nothing else, so *declined* counts a model breaking
format rather than a considered "not there" — with this prompt, a model that cannot find the
element still has to guess, and the guess lands in one of the rows above. The complementary
question, whether a model says "not found" when the element is genuinely absent, is a separate
measurement with its own set of absent-element queries; the annotator's
[92 flagged elements](annotations/flagged-by-annotator.md) are its starting point. The tree track
is the one place this benchmark does let a model decline: `NOT_FOUND` is an invited answer there,
which is why its *not answered* line is the model's own call rather than a format slip.

Every result file carries the class per row (`answer_class`, with the id of the element chosen
instead) and the counts (`summary.answer_classes`), and the
[viewer](https://dataset-review-ui-test.kobiton.com/compare) filters on them — pick *Wrong
element* and each screenshot shows the asked-for box and the chosen one side by side. The worst
cases of each kind, with those links, are listed per model:

- [GPT-5.6](error-analysis/gpt-5-6-terra.md)
- [Claude Opus 4.8](error-analysis/claude-opus-4-8.md)
- [Qwen2.5-VL-7B](error-analysis/qwen2-5-vl-7b.md)
- [GUI-Owl-1.5-8B](error-analysis/gui-owl-1-5-8b.md)

Each page is generated from the full result file, and can be regenerated from your own:

```bash
ai-testing-benchmark failures benchmark-results/vision-<model>-….json \
    --dataset data/labels/ground-truth.jsonl --annotation-origin gpt-5.6 --phrasing 0 --per-class 40 \
    > docs/error-analysis/<model>.md
```

The rule behind the classes is `_answer_classes` in
`src/ai_testing_benchmark/benchmark/artifacts/builder.py`.

## Tree track: answers stuck in a repetition loop

A tree-track answer is one XPath; the longest GPT-5.6 or Opus returned in 31,704 requests is 421
characters. Qwen2.5-VL-7B produced 24 answers (0.08%) that did not terminate: a path step such as
`/android.widget.FrameLayout[@resource-id='…']` repeated up to 60 times, or a `parent::` chain 70
deep, until the 2,048-token cap.

At the endpoint's 20–25 tokens per second one such answer takes 65–115 seconds, which is how the
run came to record client timeouts; re-asked, 6 hit the cap again, the run's 6 `error` rows. The 24
fall on 11 screens whose trees are larger than the median but inside the normal range; on the same
24 requests GPT-5.6 and Opus each answered 14 correctly, with XPaths of at most 178 characters. No
figure moves — the answers are already *not answered* or excluded — but the 24 took 9.1% of the
run's output tokens and held other requests in the queue for over a minute.

The rows, the full answers and the arithmetic are in
[`error-analysis/repetition-loops/`](error-analysis/repetition-loops/): a
[README](error-analysis/repetition-loops/README.md) with the method, a
[table of the 24](error-analysis/repetition-loops/qwen2.5-vl-7b.md) linking each screenshot and its
page source, and the full answers as JSONL.

## Tree track: what "not answered" is made of

The tree track's failures are counted in the README's tree table rather than browsed here, because
most of them are not the model's: a third of the requests describe an element the page source has
no node for — the on-screen keyboard above all, which the dumps do not contain — and the model's
`NOT_FOUND` is the right answer. The sublines of that table split the rest into an XPath that did
not compile and one that compiled but matched nothing, and the second is where the three models
differ most: 304 for GPT-5.6, 283 for Opus, 4,344 for Qwen, which names attributes the tree does
not have. [The methodology page](methodology.md#--input-xml--the-xml-track) explains each outcome.

## The ground truth's own errors

The 55 elements the annotator could not find on the screen and the 37 whose description fit more
than one thing are excluded from every figure. They are listed, with the screenshot and page source
of each, in [`annotations/flagged-by-annotator.md`](annotations/flagged-by-annotator.md); what the
adjudication found about the two model labellings is in [annotations.md](annotations.md).
