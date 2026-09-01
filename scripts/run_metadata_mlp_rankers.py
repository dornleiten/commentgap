#!/usr/bin/env python3
"""Run the metadata-only neural ranking workflow."""

from commentgap_analysis.neural_ranker_cli import main


if __name__ == "__main__":
    main(default_approach="metadata_mlp")
