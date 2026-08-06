import os

from dotenv import dotenv_values, load_dotenv

load_dotenv()

# .env wins for the AWS credentials only. A shell that already exports keys for
# an unrelated account otherwise shadows the file and every S3 call 403s.
# Everything else keeps normal precedence on purpose, so `LLM_PROVIDER=openai
# python run_pipeline.py ...` overrides the file for one run instead of being
# silently undone by it.
_FILE_VALUES = dotenv_values()
for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
    if _FILE_VALUES.get(_key):
        os.environ[_key] = _FILE_VALUES[_key]


class Config:
    # LLM provider: "openai" or "claude"
    llm_provider: str = os.getenv("LLM_PROVIDER", "claude")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Anthropic / Claude
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix_screenshots: str = os.getenv("S3_PREFIX_SCREENSHOTS", "raw-screenshots")
    s3_prefix_elements: str = os.getenv("S3_PREFIX_ELEMENTS", "extracted-elements")
    s3_dataset_key: str = os.getenv("S3_DATASET_KEY", "dataset-v1.jsonl")
    s3_stats_key: str = os.getenv("S3_STATS_KEY", "dataset-v1-stats.json")

    # Screenshots labeled in parallel, and the sibling of BENCHMARK_WORKERS. The unit
    # is a screenshot, not a request: a worker walks its own screenshot's ~16 calls one
    # after another, so this is also how many requests are open at once.
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

    def boto3_kwargs(self) -> dict:
        """Return kwargs for boto3.client() — includes explicit credentials if set."""
        kwargs = {"region_name": self.aws_region}
        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        return kwargs


config = Config()


def dataset_filename(name: str) -> str:
    """A dataset name spelled so it can be found again.

    Readers filter on the extension, so a name typed without it produces a dataset that
    writes cleanly, logs success and is then invisible to everything that lists `*.jsonl`.
    That cost us a 10,647-element dataset which sat in a bucket unusable until it was
    renamed by hand.

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
