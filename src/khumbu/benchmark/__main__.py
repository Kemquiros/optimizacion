"""Entry point: ``python -m khumbu.benchmark``."""

from khumbu.benchmark.runner import format_table, run_benchmark

if __name__ == "__main__":
    print(format_table(run_benchmark()))
