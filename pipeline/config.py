import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Everything reads through the environment with normal precedence, so
# `LLM_PROVIDER=openai python run_pipeline.py ...` overrides the .env file for one run.

# Everything the pipeline reads and writes lives under this directory, anchored on the
# repository root rather than the caller's CWD. Output used to be a bare relative path, so
# `run_pipeline.py --output-name mine` dropped mine.jsonl into whatever directory you
# happened to be standing in and a later run from elsewhere could not find its own
# checkpoint.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


class Config:
    # LLM provider: "openai" or "claude"
    llm_provider: str = os.getenv("LLM_PROVIDER", "claude")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Anthropic / Claude
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    # Screenshots labeled in parallel. The unit is a screenshot, not a request: a worker
    # walks its own screenshot's ~14 calls one after another, so this is also how many
    # requests are open at once.
    pipeline_workers: int = int(os.getenv("PIPELINE_WORKERS", "15"))

    @property
    def active_model(self) -> str:
        """The model `llm_client.call_llm` will actually use.

        Mirrors that function's own provider dispatch — one `if` in both places, kept
        together on purpose: step 4 records this in the dataset's stats sidecar, and a
        recorded model that disagrees with the one that did the labelling is worse than
        no record at all.
        """
        return (self.anthropic_model if self.llm_provider.lower() == "claude"
                else self.openai_model)


config = Config()


def dataset_filename(name: str) -> str:
    """A dataset name spelled so it can be found again.

    Readers filter on the extension, so a name typed without it produces a dataset that
    writes cleanly, logs success and is then invisible to everything that lists `*.jsonl`.
    That cost us a 10,647-element dataset which sat unusable until it was renamed by hand.

    Normalizing in one place rather than at each call site is what keeps the output file,
    the stats sidecar and the checkpoint agreeing on a single spelling.
    """
    return name if name.endswith(".jsonl") else name + ".jsonl"


def stats_filename(name: str) -> str:
    """The stats sidecar belonging beside `name`.

    Deliberately not `Path(name).stem`, which strips whatever follows the *last* dot: a
    version number in the name loses part of itself, so a dataset called `run-gpt-5.6-alpha`
    produced `run-gpt-5-stats.json` — a sidecar that no longer names its own dataset. Going
    through `dataset_filename` first means the two are derived from the same string whether
    or not the caller typed the extension.
    """
    return dataset_filename(name).removesuffix(".jsonl") + "-stats.json"
