"""Back-compat entry point. Real implementation lives in cli.py.

Both invocations work:
    python -m baselines.tinker_experiments.run_experiments [args]
    python -m baselines.tinker_experiments.cli            [args]
"""
from .cli import main

if __name__ == "__main__":
    main()
