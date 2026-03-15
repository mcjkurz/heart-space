#!/usr/bin/env python3
"""
Train separate Word2Vec models per period with multi-trial support.

Supports training multiple models per period with different random seeds,
parallel execution, and skip-if-exists logic.

Usage (run from project root):
  # Default: train 1 model per period (3 epochs, seed 42)
  python scripts/train_period_models.py

  # Train 10 models per period, 4 parallel processes, 5 epochs each
  python scripts/train_period_models.py --trials 10 --processes 4 --epochs 5

  # Custom output directory
  python scripts/train_period_models.py --trials 10 --output-dir models/period_models/
"""

import argparse
import os
import time
from multiprocessing import Pool
from typing import Dict, Iterable, List

from qhchina.analytics.word2vec import Word2Vec
from qhchina.utils import LineSentenceFile


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']

MODEL_PARAMS = {
    "vector_size": 200,
    "window": 10,
    "min_word_count": 3,
    "sg": 1,
    "negative": 10,
    "alpha": 0.025,
    "seed": 42,
    "epochs": 3,
    "calculate_loss": False,
    "workers": 4,
}


class ReplacingLineSentenceFile:
    """Wrapper around LineSentenceFile that replaces target words on-the-fly."""

    def __init__(self, filepath: str, replacements: Dict[str, str]):
        self.filepath = filepath
        self.replacements = replacements

    def __iter__(self) -> Iterable[List[str]]:
        for sentence in LineSentenceFile(self.filepath):
            yield [self.replacements.get(word, word) for word in sentence]


def load_target_words(words_file: str) -> List[str]:
    """Load target words from a text file (one word per line)."""
    if not os.path.exists(words_file):
        raise FileNotFoundError(
            f"Target words file not found: {words_file}\n"
            f"Expected a text file with one word per line."
        )
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


def load_period_sentences(data_dir: str, replacements: Dict[str, str] = None):
    """Load segmented sentences from per-period .txt files with optional replacement."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            if replacements:
                corpora[period] = ReplacingLineSentenceFile(filepath, replacements)
            else:
                corpora[period] = LineSentenceFile(filepath)
    return corpora


def make_model_filename(period: str, epochs: int, seed: int) -> str:
    """Generate model filename: {period}_e{epochs}_s{seed}.npy"""
    return f"{period}_e{epochs}_s{seed}.npy"


def train_single_model(args: tuple) -> str:
    """
    Train a single Word2Vec model for one period and save it.

    Top-level function for multiprocessing compatibility.
    Returns a status message string.
    """
    (period, seed, data_dir, output_path, params, replacements) = args

    trial_params = params.copy()
    trial_params["seed"] = seed

    filepath = os.path.join(data_dir, f"sentences_{period}.txt")
    if replacements:
        sentences = ReplacingLineSentenceFile(filepath, replacements)
    else:
        sentences = LineSentenceFile(filepath)

    model = Word2Vec(sentences=sentences, **trial_params)
    model.train()
    model.save(output_path)

    return f"Saved {output_path} (period={period}, seed={seed})"


def main():
    parser = argparse.ArgumentParser(
        description="Train per-period Word2Vec models with multi-trial support."
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of models to train per period (default: 1)"
    )
    parser.add_argument(
        "--processes", type=int, default=1,
        help="Number of parallel processes for training (default: 1)"
    )
    parser.add_argument(
        "--output-dir", default="models/period_models",
        help="Directory for saved .npy model files (default: models/period_models)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Training epochs per model (default: 3)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=42,
        help="Starting seed; trial i uses seed_start + i (default: 42)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-train models even if output file already exists"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
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
    parser.add_argument(
        "--min-count", type=int, default=None,
        help="Minimum word count for model vocabulary (default: 3)"
    )

    args = parser.parse_args()

    # Build model params
    params = MODEL_PARAMS.copy()
    params["epochs"] = args.epochs
    if args.vector_size is not None:
        params["vector_size"] = args.vector_size
    if args.window is not None:
        params["window"] = args.window
    if args.min_count is not None:
        params["min_word_count"] = args.min_count

    # Load target words and build replacements
    target_words = load_target_words(args.words)
    replacements = {word: args.replacement for word in target_words}
    print(f"Loaded {len(target_words)} target words, replacement: '{args.replacement}'")

    # Discover available periods
    available_periods = []
    for period in PERIODS:
        filepath = os.path.join(args.data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            available_periods.append(period)
    print(f"Found {len(available_periods)} periods: {available_periods}")

    # Build seed list and output paths, skipping existing models
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [args.seed_start + i for i in range(args.trials)]
    jobs = []
    skipped = 0

    for period in available_periods:
        for seed in seeds:
            filename = make_model_filename(period, args.epochs, seed)
            output_path = os.path.join(args.output_dir, filename)
            if os.path.exists(output_path) and not args.overwrite:
                skipped += 1
                continue
            jobs.append((period, seed, args.data_dir, output_path, params, replacements))

    if skipped > 0:
        print(f"Skipping {skipped} already-trained model(s) (use --overwrite to re-train)")

    if not jobs:
        print("All models already exist. Nothing to do.")
        return

    # Print summary
    print(f"\nModels to train: {len(jobs)} ({args.trials} trial(s) x {len(available_periods)} period(s))")
    print(f"Seeds: {seeds[0]} to {seeds[-1]}")
    print(f"Epochs: {args.epochs}")
    print(f"Processes: {args.processes}")
    print(f"Output: {args.output_dir}")
    print(f"Filename pattern: {{period}}_e{args.epochs}_s{{seed}}.npy")
    print(f"Model params: {params}")
    print()

    # Dispatch training
    def fmt_elapsed(start: float) -> str:
        elapsed = int(time.time() - start)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    t0 = time.time()

    if args.processes <= 1 or len(jobs) == 1:
        for i, task in enumerate(jobs, 1):
            period, seed = task[0], task[1]
            print(f"[{i}/{len(jobs)}] Training {period} seed {seed}...")
            result = train_single_model(task)
            print(f"  {result} [{fmt_elapsed(t0)} elapsed]")
    else:
        print(f"Launching {args.processes} parallel processes...")
        with Pool(processes=args.processes) as pool:
            for i, result in enumerate(pool.imap_unordered(train_single_model, jobs), 1):
                print(f"  [{i}/{len(jobs)}] {result} [{fmt_elapsed(t0)} elapsed]")

    total_elapsed = fmt_elapsed(t0)
    print(f"\nDone! {len(jobs)} model(s) saved to {args.output_dir}/ [total: {total_elapsed}]")


if __name__ == "__main__":
    main()

