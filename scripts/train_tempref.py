#!/usr/bin/env python3
"""
Train a TempRefWord2Vec model for interiority semantic change analysis.

Loads segmented sentence data from per-period .txt files, replaces target words
with a unified token, and trains a TempRefWord2Vec model.

Usage (run from project root):
  python scripts/train_tempref.py
  python scripts/train_tempref.py --data-dir data/segmented --output models/tempref_interiority_w2v.npy
  python scripts/train_tempref.py --words data/dictionaries/interiority_words.txt --epochs 5
  python scripts/train_tempref.py --replacement my_concept
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Set

from qhchina.analytics import TempRefWord2Vec
from qhchina.utils import LineSentenceFile


class ReplacingLineSentenceFile:
    """Wrapper around LineSentenceFile that replaces target words on-the-fly."""

    def __init__(self, filepath: str, replacements: Dict[str, str]):
        """
        Args:
            filepath: Path to the sentence file.
            replacements: Dict mapping original words to replacement tokens.
        """
        self.filepath = filepath
        self.replacements = replacements

    def __iter__(self) -> Iterable[List[str]]:
        for sentence in LineSentenceFile(self.filepath):
            yield [self.replacements.get(word, word) for word in sentence]


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']

MODEL_PARAMS = {
    "vector_size": 200,
    "window": 10,
    "min_word_count": 3,
    "epochs": 3,
    "sg": 1,
    "negative": 10,
    "alpha": 0.025,
    "seed": 42,
    "calculate_loss": True,
    "workers": 4,
    "sampling_strategy": "balanced",
}


def load_target_words(words_file):
    """Load target words from a text file (one word per line)."""
    if not os.path.exists(words_file):
        raise FileNotFoundError(
            f"Target words file not found: {words_file}\n"
            f"Expected a text file with one word per line at data/dictionaries/interiority_words.txt"
        )
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


def load_period_sentences(data_dir: str, replacements: Dict[str, str] = None) -> Dict[str, Iterable[List[str]]]:
    """Load segmented sentences from per-period .txt files with optional word replacement."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            if replacements:
                corpora[period] = ReplacingLineSentenceFile(filepath, replacements)
            else:
                corpora[period] = LineSentenceFile(filepath)
            print(f"  {period}: {filepath}")
        else:
            print(f"  {period}: (not found)")
    print(f"Loaded {len(corpora)} periods: {list(corpora.keys())}")
    return corpora


def main():
    parser = argparse.ArgumentParser(
        description="Train a TempRefWord2Vec model for semantic change analysis."
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--output", default="models/tempref_interiority_w2v.npy",
        help="Output path for the trained model"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file (one word per line)"
    )
    parser.add_argument(
        "--replacement", default="interiority",
        help="Replacement token for target words (default: interiority)"
    )
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing model without prompting"
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        response = input(
            f"Model already exists at {args.output}. Overwrite? [y/N] "
        ).strip().lower()
        if response not in ("y", "yes"):
            print("Aborted. Exiting.")
            sys.exit(0)

    params = MODEL_PARAMS.copy()
    if args.vector_size is not None:
        params["vector_size"] = args.vector_size
    if args.window is not None:
        params["window"] = args.window
    if args.min_count is not None:
        params["min_word_count"] = args.min_count
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.workers is not None:
        params["workers"] = args.workers
    if args.seed is not None:
        params["seed"] = args.seed

    target_words = load_target_words(args.words)
    print(f"Using {len(target_words)} target words, replacement token: '{args.replacement}'")

    replacements = {word: args.replacement for word in target_words}

    print(f"\nLoading sentences from {args.data_dir}...")
    corpora = load_period_sentences(args.data_dir, replacements=replacements)

    print(f"\nTraining TempRefWord2Vec model (params: {params})...")
    model = TempRefWord2Vec(
        sentences=corpora,
        targets=[args.replacement],
        **params,
    )
    model.train()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Labels: {model.labels}")
    print(f"Targets: {model.targets}")


if __name__ == "__main__":
    main()
