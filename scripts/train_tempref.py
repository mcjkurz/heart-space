#!/usr/bin/env python3
"""
Train one or more TempRefWord2Vec models for semantic change analysis.

Supports multi-trial training with different random seeds, parallel execution,
and optional period permutation (for null distribution models).

Usage (run from project root):
  # Default: train 1 model (3 epochs, seed 42)
  python scripts/train_tempref.py

  # Train 100 models, 4 parallel processes, 5 epochs each
  python scripts/train_tempref.py --trials 100 --processes 4 --epochs 5 --output-dir models/real/

  # Train 100 null (permuted-period) models
  python scripts/train_tempref.py --trials 100 --processes 4 --epochs 5 --permute-periods --output-dir models/null/
"""

import argparse
import os
import random
import shutil
import tempfile
import time
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple

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
    "calculate_loss": False,
    "workers": 4,
    "sampling_strategy": "balanced",
}


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


def load_period_sentences(
    data_dir: str, replacements: Dict[str, str]
) -> Dict[str, ReplacingLineSentenceFile]:
    """Load segmented sentences from per-period .txt files with word replacement."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            corpora[period] = ReplacingLineSentenceFile(filepath, replacements)
    return corpora


def build_sentence_index(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """Build global sentence index: [(period, line_idx), ...] and period sizes."""
    sentence_index = []
    period_sizes = {}

    print("Building sentence index for permutation...")
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                n_lines = sum(1 for _ in f)
            sentence_index.extend([(period, i) for i in range(n_lines)])
            period_sizes[period] = n_lines
            print(f"  {period}: {n_lines:,} sentences")

    print(f"  Total: {len(sentence_index):,} sentences")
    return sentence_index, period_sizes


def create_permuted_corpus_files(
    data_dir: str,
    temp_dir: str,
    sentence_index: List[Tuple[str, int]],
    period_sizes: Dict[str, int],
    seed: int
) -> None:
    """Create temporary permuted sentence files in temp_dir."""
    rng = random.Random(seed)
    shuffled_index = sentence_index.copy()
    rng.shuffle(shuffled_index)

    assignments = {}
    idx = 0
    for period in PERIODS:
        if period not in period_sizes:
            continue
        size = period_sizes[period]
        assignments[period] = shuffled_index[idx:idx + size]
        idx += size

    source_files = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                source_files[period] = f.readlines()

    for period, assigned_indices in assignments.items():
        temp_path = os.path.join(temp_dir, f"sentences_{period}.txt")
        with open(temp_path, 'w', encoding='utf-8') as f:
            for src_period, line_idx in assigned_indices:
                f.write(source_files[src_period][line_idx])


def train_single_model(args: tuple) -> str:
    """
    Train a single TempRefWord2Vec model and save it.

    This is a top-level function (required for multiprocessing pickling).
    Accepts a single tuple argument for compatibility with Pool.map().

    Returns:
        A status message string.
    """
    (
        seed, data_dir, output_path, params, replacements, target,
        permute, sentence_index, period_sizes
    ) = args

    trial_params = params.copy()
    trial_params["seed"] = seed

    if permute:
        temp_dir = tempfile.mkdtemp(prefix="permuted_corpus_")
        try:
            create_permuted_corpus_files(
                data_dir, temp_dir, sentence_index, period_sizes, seed
            )
            corpora = load_period_sentences(temp_dir, replacements)
            model = TempRefWord2Vec(
                sentences=corpora,
                targets=[target],
                **trial_params,
            )
            model.train()
            model.save(output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        corpora = load_period_sentences(data_dir, replacements)
        model = TempRefWord2Vec(
            sentences=corpora,
            targets=[target],
            **trial_params,
        )
        model.train()
        model.save(output_path)

    return f"Saved {output_path} (seed={seed})"


def make_model_filename(epochs: int, seed: int, permute: bool) -> str:
    """Generate model filename following the e{epochs}_s{seed} convention."""
    suffix = "_null" if permute else ""
    return f"model_e{epochs}_s{seed}{suffix}.npy"


def main():
    parser = argparse.ArgumentParser(
        description="Train one or more TempRefWord2Vec models for semantic change analysis."
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of models to train (default: 1)"
    )
    parser.add_argument(
        "--processes", type=int, default=1,
        help="Number of parallel processes for training models (default: 1)"
    )
    parser.add_argument(
        "--output-dir", default="./models/",
        help="Directory for saved .npy model files (default: ./models/)"
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
        "--permute-periods", action="store_true",
        help="Shuffle sentences across period labels before training (null distribution mode)"
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
    parser.add_argument("--min-count", type=int, default=None,
                        help="Minimum word count for model vocabulary (default: 3)")

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

    # Build seed list and output paths, skipping existing models
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [args.seed_start + i for i in range(args.trials)]
    jobs = []
    skipped = 0

    for seed in seeds:
        filename = make_model_filename(args.epochs, seed, args.permute_periods)
        output_path = os.path.join(args.output_dir, filename)
        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue
        jobs.append(seed)

    if skipped > 0:
        print(f"Skipping {skipped} already-trained model(s) (use --overwrite to re-train)")

    if not jobs:
        print("All models already exist. Nothing to do.")
        return

    # Build sentence index if permuting
    sentence_index = None
    period_sizes = None
    if args.permute_periods:
        sentence_index, period_sizes = build_sentence_index(args.data_dir)

    # Print summary
    mode = "null distribution (permuted periods)" if args.permute_periods else "real (original periods)"
    print(f"\nMode: {mode}")
    print(f"Models to train: {len(jobs)} (of {args.trials} total)")
    print(f"Seeds: {jobs[0]} to {jobs[-1]}")
    print(f"Epochs: {args.epochs}")
    print(f"Processes: {args.processes}")
    print(f"Output: {args.output_dir}")
    print(f"Filename pattern: {make_model_filename(args.epochs, '<seed>', args.permute_periods)}")
    print(f"Model params: {params}")
    print()

    # Build task arguments
    task_args = []
    for seed in jobs:
        filename = make_model_filename(args.epochs, seed, args.permute_periods)
        output_path = os.path.join(args.output_dir, filename)
        task_args.append((
            seed, args.data_dir, output_path, params, replacements,
            args.replacement, args.permute_periods, sentence_index, period_sizes
        ))

    # Dispatch training
    def fmt_elapsed(start: float) -> str:
        """Format elapsed time as HH:MM:SS."""
        elapsed = int(time.time() - start)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    t0 = time.time()

    if args.processes <= 1 or len(task_args) == 1:
        # Sequential mode
        for i, task in enumerate(task_args, 1):
            print(f"[{i}/{len(task_args)}] Training seed {task[0]}...")
            result = train_single_model(task)
            print(f"  {result} [{fmt_elapsed(t0)} elapsed]")
    else:
        # Parallel mode
        print(f"Launching {args.processes} parallel processes...")
        with Pool(processes=args.processes) as pool:
            for i, result in enumerate(pool.imap_unordered(train_single_model, task_args), 1):
                print(f"  [{i}/{len(task_args)}] {result} [{fmt_elapsed(t0)} elapsed]")

    total_elapsed = fmt_elapsed(t0)
    print(f"\nDone! {len(task_args)} model(s) saved to {args.output_dir}/ [total: {total_elapsed}]")


if __name__ == "__main__":
    main()
