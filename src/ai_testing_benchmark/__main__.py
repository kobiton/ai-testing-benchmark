"""`python -m ai_testing_benchmark …` - the same entry point as the `ai-testing-benchmark` script."""
import sys

from ai_testing_benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
