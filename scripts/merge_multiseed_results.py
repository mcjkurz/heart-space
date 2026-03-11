#!/usr/bin/env python3
"""
Merge multiple multiseed/null distribution results from parallel runs.

Usage:
  python scripts/merge_multiseed_results.py results/null_part* --output results/null_distribution
  python scripts/merge_multiseed_results.py results/multiseed_part* --output results/multiseed --null-dir results/null_distribution
"""

import argparse
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from semantic_change_multiseed import (
    aggregate_and_save_results,
    compute_cooccurrences,
    compute_word_counts,
    load_target_words,
    PERIODS,
)


def merge_npz_files(input_dirs: list, output_path: str):
    """Merge multiple .npz checkpoint files into one."""
    merged = defaultdict(lambda: defaultdict(list))
    
    for input_dir in input_dirs:
        npz_files = glob(os.path.join(input_dir, "*.npz"))
        if not npz_files:
            print(f"  Warning: No .npz files found in {input_dir}")
            continue
        
        for npz_file in npz_files:
            print(f"  Loading {npz_file}...")
            data = np.load(npz_file, allow_pickle=True)
            for key in data.files:
                transition, word = key.split("|", 1)
                merged[transition][word].extend(data[key].tolist())
    
    save_dict = {}
    for transition, word_scores in merged.items():
        for word, scores in word_scores.items():
            key = f"{transition}|{word}"
            save_dict[key] = np.array(scores)
    
    np.savez_compressed(output_path, **save_dict)
    
    total_trials = 0
    if merged:
        first_word = next(iter(next(iter(merged.values())).values()))
        total_trials = len(first_word)
    
    return merged, total_trials


def main():
    parser = argparse.ArgumentParser(
        description="Merge parallel multiseed results and regenerate CSVs."
    )
    parser.add_argument(
        "input_dirs", nargs="+",
        help="Input directories containing partial results (e.g., results/null_part1 results/null_part2)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for merged results"
    )
    parser.add_argument(
        "--null-dir", default=None,
        help="Path to null distribution for p-value calculation (for multiseed mode)"
    )
    parser.add_argument(
        "--is-null", action="store_true",
        help="Set if merging null distribution results"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing segmented sentence files"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file"
    )
    parser.add_argument(
        "--min-word-length", type=int, default=2
    )
    parser.add_argument(
        "--min-cooc", type=int, default=5
    )
    parser.add_argument(
        "--min-count", type=int, default=5
    )
    parser.add_argument(
        "--postag", default="n.*"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    checkpoint_file = "null_scores.npz" if args.is_null else "multiseed_scores.npz"
    output_npz = os.path.join(args.output, checkpoint_file)
    
    print(f"Merging {len(args.input_dirs)} directories...")
    merged, total_trials = merge_npz_files(args.input_dirs, output_npz)
    print(f"Merged {total_trials} trials, saved to {output_npz}")
    
    print("\nComputing word counts and co-occurrences...")
    word_counts = compute_word_counts(args.data_dir)
    
    target_words = load_target_words(args.words)
    cooc_count = None
    if args.min_cooc > 0:
        cooc_count = compute_cooccurrences(args.data_dir, target_words)
    
    postag = args.postag if args.postag.lower() != 'none' else None
    
    print("\nGenerating CSVs...")
    aggregate_and_save_results(
        merged, args.output,
        is_null_mode=args.is_null,
        null_dir=args.null_dir,
        word_counts=word_counts,
        cooc_count=cooc_count,
        min_word_length=args.min_word_length,
        min_cooc=args.min_cooc,
        min_count=args.min_count,
        postag=postag
    )
    
    print(f"\nDone! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
