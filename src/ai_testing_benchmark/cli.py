"""`ai-testing-benchmark <command> …` - one entry point for the whole package.

    run        score a model on the dataset: the screenshot track or the tree track
    cascade    join an xml- and a vision- result into the tree-then-screenshot cascade
    failures   write a result's failures, class by class, as a Markdown page
    pipeline   label your own screenshots into a dataset

Each command keeps its own flags (`ai-testing-benchmark run --help`) and takes `--config` for the JSON form of them;
see `config.py` and the `config/` directory.
Commands are imported on demand, so `run` works without the pipeline's dependencies installed.
"""
import importlib
import importlib.util
import sys

from ai_testing_benchmark import __version__

COMMANDS = {
    "run": ("ai_testing_benchmark.benchmark.cli.commands", "score a model on the dataset (screenshot or tree track)"),
    "cascade": ("ai_testing_benchmark.benchmark.cascade_report", "tree-then-screenshot cascade from an xml- and a vision- result"),
    "failures": ("ai_testing_benchmark.benchmark.failures_report", "a result's failures, class by class, as Markdown"),
    "pipeline": ("ai_testing_benchmark.pipeline.cli", "label your own screenshots into a dataset"),
}
# Commands a checkout may carry beyond the published set; registered only when importable.
OPTIONAL_COMMANDS = {
    "outcomes": ("ai_testing_benchmark.benchmark.outcomes", "classify a result's failures by outcome"),
}

PROG = "ai-testing-benchmark"


def _usage(commands: dict) -> str:
    width = max(len(name) for name in commands)
    lines = [f"usage: {PROG} <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_, summary) in commands.items()]
    lines += ["", f"`{PROG} <command> --help` lists that command's options; `--config FILE` takes them from JSON."]
    return "\n".join(lines)


def _available() -> dict:
    commands = dict(COMMANDS)
    for name, (module, summary) in OPTIONAL_COMMANDS.items():
        if importlib.util.find_spec(module) is not None:
            commands[name] = (module, summary)
    return commands


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = _available()
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(commands))
        return 0
    if argv[0] == "--version":
        print(f"{PROG} {__version__}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in commands:
        print(f"{PROG}: unknown command '{name}'\n\n{_usage(commands)}", file=sys.stderr)
        return 2
    try:
        module = importlib.import_module(commands[name][0])
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("ai_testing_benchmark"):
            raise
        extra = ' ".[pipeline]"' if name == "pipeline" else " ."
        print(f"{PROG} {name}: missing dependency '{e.name}' - install it with `pip install -e{extra}`", file=sys.stderr)
        return 2
    code = module.main(rest)
    return 0 if code is None else int(code)


if __name__ == "__main__":
    sys.exit(main())
