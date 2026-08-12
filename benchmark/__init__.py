"""Vision benchmark for natural-language element locators.

Layered, and the layers only depend downwards:

    cli/        argument parsing and dispatch
    engine/     the run — thread pool, per-element step, stop handling
    artifacts/  checkpoint, result JSON, re-scoring
    model/      the system under test — prompt, image encoding, HTTP
    scoring/    coordinate conventions, response parsing, metrics
    workload/   the dataset rows, phrasings, and --limit sampling

`workload` and `scoring` depend on nothing internal; `model` on `scoring`;
`artifacts` on `scoring`, `model` and `workload`; `engine` on all of those;
`cli` on `engine` and `artifacts`. An import pointing the other way is what
would make this circular, so add one only after checking.
"""
