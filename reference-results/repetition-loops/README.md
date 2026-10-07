# Answers stuck in a repetition loop on the tree track

A tree-track request is small: the filtered page source and one description go in, one XPath or `NOT_FOUND` comes out. 

GPT-5.6 and Claude Opus 4.8 answered all 31,704 requests that way; the longest XPath either returned is 421 characters.

Qwen2.5-VL-7B answered 31,680 of them that way too (median 10 tokens), and on **24 requests (0.08%) did not terminate** 
before the token cap or the client timeout: 19 of them stuck in a repetition loop, 5 in long prose. 

This folder lists those 24 and measures their cost.

Files:

- [`qwen2.5-vl-7b.md`](qwen2.5-vl-7b.md): the 24 rows one by one, what was asked, what the model wrote, how long it took, 
  and what GPT-5.6 and Opus answered on the same tree and description.
- [`qwen2.5-vl-7b.jsonl`](qwen2.5-vl-7b.jsonl): the same rows with the full answer text and the other two models' XPaths.

## Form of the answers

Asked for *the three-dot button beside Ian Dunt's post* ([png](../../data/images/d6461ce91259efa8732e195a9767ddaf67d0cf6357eef144f33558842cedca97.png) · [xml](../../data/xml/d6461ce91259efa8732e195a9767ddaf67d0cf6357eef144f33558842cedca97.xml)), the model wrote

```
//android.widget.TextView[text()='…see more']/parent::android.view.View/parent::android.view.View/parent::android.view.View/...
```

and appended `/parent::android.view.View` until the 2,048-token cap: 8,724 characters in 86 seconds. 

GPT-5.6 answered the same request with a 114-character XPath. 

Asked for *the center SuperDeals product showing a vacuum* ([png](../../data/images/f0a8759a97f90e5099681cb207dc88f6431d7244bac9880ac7ee2a4cd35dd8ce.png) · [xml](../../data/xml/f0a8759a97f90e5099681cb207dc88f6431d7244bac9880ac7ee2a4cd35dd8ce.xml)), it repeated `/android.widget.FrameLayout[@resource-id='...']` 
for 1,534 tokens and 67 seconds; GPT-5.6 found the element with a 178-character XPath.

| form                                                                                                  | count |
|-------------------------------------------------------------------------------------------------------|-------|
| one child step repeated 11–60 times (`/LinearLayout[@resource-id='…card_content']/LinearLayout[…]/…`) | 18    |
| a `parent::` chain, 70 steps                                                                          | 1     |
| long free text with no single XPath                                                                   | 5     |

There is no mild form. Every one of the other 31,680 answers is a normal XPath or `NOT_FOUND`; no answer has between 3
and 20 `parent::` steps. Answers are either ordinary or unterminated.

## Distribution

11 screens, with trees larger than the median but inside the range of the whole set:

|                                             | the 11 screens | all 841 screens, median / p90 |
|---------------------------------------------|----------------|-------------------------------|
| nodes in the tree                           | 68 – 298       | 52 / 113                      |
| depth                                       | 16 – 36        | 16 / 25                       |
| siblings of the same class under one parent | 3 – 12         | 4 / 10                        |

Four screens account for 16 of the 24 rows; on three of them every phrasing of the same element ran away. 
By phrasing: `name` 9, `label` 10, `intent` 5.

On the same 24 (tree, description) pairs GPT-5.6 answered 14 correctly and Opus 14; their longest XPaths there were 178 and 224 characters.

## Cost

The endpoint generates 20–25 tokens per second. A capped answer therefore takes 65–115 seconds on its own, and the run's
client timeout is 120 seconds, so an answer that reaches the cap while four other requests share the server crosses it.
That is what the first pass of the run recorded: **18 errors, 10 of them timeouts**, the rest answers cut by the cap.
A resume re-asked the 18; 12 stopped in time, 6 hit the cap again and are the 6 `error` rows of the final file.

Beside the error count:

- The 18 unterminated answers that did stop held the generator for 15 minutes of a 106-minute run (5 streams); 
  with the 6 capped and 10 timed-out ones, roughly 45 minutes of stream time went to 34 requests.
- 9.1% of all output tokens of the run (32,438 of 358,389) are in 24 answers.
- Five ordinary 3-token answers on another screen waited 64–68 seconds, queued behind an unterminated one. 
  The run's median latency is 0.7 s and its p99 2.9 s.

## Effect on the reported figures

Nothing. The 6 capped answers are `error` rows and excluded from every figure; 
the 18 that finished are `invalid_xpath` or `no_match` and counted as *not answered*, which is where an unusable locator lands whatever its length. 
A `--max-tokens 512` on a future run would end these answers in about 20 seconds instead of 100, and keep them from holding up other requests, 
without changing a single verdict: the longest XPath any model returned on this track is 421 characters, about 140 tokens. 
(The cap is for this endpoint. GPT-5.6's recorded `output_tokens` reach 1,118 on a 107-character answer 
because the figure includes the model's hidden reasoning tokens; a cap on that model has to cover those too.)

## Method

From the result file and its checkpoint: a row is listed when `output_tokens` exceeds 300 or `error` says the answer was truncated at `max_tokens`. 
GPT-5.6 and Opus columns come from the two `xml-…` result files on the same key (screenshot, element, phrasing). 
Tree sizes are measured on the full dumps in `data/xml/`.
