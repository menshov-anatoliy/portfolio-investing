#!/usr/bin/env python
"""Thin wrapper so the script can be run directly as ``python scripts/run_experiment.py``.

The actual implementation lives in ``portfolio_investing.cli``, which is also
registered as the ``run-experiment`` console-script entry point in pyproject.toml.
"""

from portfolio_investing.cli import main

if __name__ == "__main__":
    main()
