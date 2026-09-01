#!/usr/bin/env python3
"""Run the frozen BGE-M3 plus metadata preference rankers."""

from commentgap_analysis.neural_ranker_cli import main


if __name__ == "__main__":
    main(default_approach="frozen_bge")
