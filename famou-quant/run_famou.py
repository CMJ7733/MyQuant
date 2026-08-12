#!/usr/bin/env python3
"""
Convenience wrapper for running Famou experiments.

This is a thin wrapper around famou.main for backward compatibility.
Preferred usage is: python -m famou [args]

Usage:
    python run_famou.py -c CONFIG -p PROGRAMS -e EVALUATOR
    python run_famou.py --resume EXPERIMENT_PATH -e EVALUATOR
"""


from famou.main import main

if __name__ == "__main__":
    main()
