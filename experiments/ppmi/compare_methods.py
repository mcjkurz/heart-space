#!/usr/bin/env python3
"""
Combine semantic-change results from multiple directories, keeping only words
that appear in the top-N of ALL given directories for each transition.

This is the cross-method agreement filter: a word must rank in the top-N
of every input directory's results to be kept. Words that survive this
filter are the strongest candidates for genuine semantic change, since they
are flagged by multiple independent methods with different theoretical
assumptions.

The CSV format may differ slightly across directories (PPMI has
change_score/sim_pre/sim_post; Word2Vec has mean_change/std_change/z_score/
p_value; etc.). This script handles that by:
  - Using `word` as the shared key.
  - Prefixing every other column with a label derived from the directory
    name, so columns from different directories never collide.

Usage (run from experiments/ppmi; the default output directory is under the repo root):
  python compare_methods.py ../../results/ppmi ../../results/multiseed
  python compare_methods.py ../../results/ppmi ../../results/multiseed --topn 50
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def transition_names() -> List[str]:
    """Return the 4 adjacent-transition names in order."""
    return [f"{PERIODS[i]}_to_{PERIODS[i+1]}" for i in range(len(PERIODS) - 1)]


def dir_label(dir_path: str) -> str:
    """Derive a short label from a directory path for column prefixing.

    Uses the basename of the directory (e.g. 'ppmi', 'multiseed').
    If basenames collide, appends a numeric suffix.
    """
    return os.path.basename(os.path.normpath(dir_path))


def load_top_words(csv_path: str, topn: int) -> Dict[str, dict]:
    """Load a semantic-change CSV and return the top-N words as a dict keyed by word.

    Args:
        csv_path: Path to the CSV file.
        topn: Number of top words to keep (CSV is assumed sorted by its
            primary score column descending).

    Returns:
        Dict mapping word -> row dict (all columns as strings).
        Empty dict if file is missing.
    """
    if not os.path.exists(csv_path):
        return {}

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Take the first topn rows (CSV is already sorted by the primary score column)
    top_rows = rows[:topn]
    return {row['word']: row for row in top_rows}


def merge_rows(word: str, labeled_rows: List[Tuple[str, dict]]) -> dict:
    """Merge rows from multiple directories into a single combined row.

    Args:
        word: The shared word key.
        labeled_rows: List of (label, row_dict) tuples from each directory.

    Returns:
        Dict with `word` plus every other column prefixed by its directory's label.
    """
    merged = {'word': word}
    for label, row in labeled_rows:
        for col, val in row.items():
            if col == 'word':
                continue
            merged[f'{label}_{col}'] = val
    return merged


def write_combined_csv(csv_path: str, rows: List[dict], all_fieldnames: List[str]) -> None:
    """Write the combined rows to a CSV."""
    if not rows:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("word\n")
            f.write("# no words in intersection for this transition\n")
        return

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in rows:
            # Ensure all fields exist
            for k in all_fieldnames:
                if k not in row:
                    row[k] = ''
            writer.writerow({k: row[k] for k in all_fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Combine semantic-change results from multiple directories, "
                     "keeping only words in the top-N of ALL directories."
    )
    parser.add_argument("dirs", nargs="+",
                        help="Two or more directories containing semantic_changes_*.csv files")
    parser.add_argument("--topn", type=int, default=30,
                        help="Top-N cutoff for each directory (default: 30)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/combined_top{topn}/)")
    parser.add_argument("--sort-by", default=None,
                        help="Column name to sort output by (descending). "
                              "Default: first directory's first score column. "
                              "Use 'word' for alphabetical sort.")
    args = parser.parse_args()

    if len(args.dirs) < 2:
        parser.error("provide at least two directories to compare")

    if args.topn < 1:
        parser.error("--topn must be >= 1")

    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT, "results", f"combined_top{args.topn}")

    # Validate directories and derive labels
    labels = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)
        labels.append(dir_label(d))

    # Ensure unique labels (append suffix if collision)
    seen = {}
    unique_labels = []
    for label in labels:
        if label in seen:
            seen[label] += 1
            unique_labels.append(f"{label}_{seen[label]}")
        else:
            seen[label] = 0
            unique_labels.append(label)
    labels = unique_labels

    os.makedirs(args.output_dir, exist_ok=True)

    transitions = transition_names()
    summary_rows = []

    print(f"Comparing top-{args.topn} across {len(args.dirs)} directories:")
    for d, l in zip(args.dirs, labels):
        print(f"  {d}  (label: '{l}')")
    print(f"Output: {args.output_dir}/")
    print()

    # Determine sort column: first directory's first non-word column by default
    sort_col = args.sort_by

    for transition in transitions:
        # Load top-N from each directory
        tops = []  # list of (label, dict_of_word_to_row)
        first_csv_path = None
        for d, label in zip(args.dirs, labels):
            csv_path = os.path.join(d, f"semantic_changes_{transition}.csv")
            top = load_top_words(csv_path, args.topn)
            tops.append((label, top))
            if first_csv_path is None and os.path.exists(csv_path):
                first_csv_path = csv_path

        # Find intersection across all directories
        word_sets = [set(t.keys()) for _, t in tops]
        if not word_sets:
            print(f"  {transition}: no data in any directory, skipping")
            summary_rows.append((transition, [0] * len(args.dirs), 0))
            continue

        common_words = set.intersection(*word_sets)

        # Build merged rows
        merged_rows = []
        for word in common_words:
            labeled_rows = [(label, top[word]) for label, top in tops if word in top]
            merged_rows.append(merge_rows(word, labeled_rows))

        # Determine sort column if not specified
        if sort_col is None and merged_rows:
            # Use the first directory's first non-word column
            first_label = labels[0]
            for col in ['change_score', 'mean_change']:
                candidate = f"{first_label}_{col}"
                if candidate in merged_rows[0]:
                    sort_col = candidate
                    break
            if sort_col is None:
                sort_col = 'word'

        # Sort by chosen column descending (or alphabetically for 'word')
        if sort_col == 'word':
            merged_rows.sort(key=lambda r: r.get('word', ''))
        else:
            merged_rows.sort(key=lambda r: float(r.get(sort_col, '0') or '0'),
                              reverse=True)

        # Collect all fieldnames (word + every prefixed column seen)
        all_fields = ['word']
        for row in merged_rows:
            for k in row.keys():
                if k != 'word' and k not in all_fields:
                    all_fields.append(k)

        out_csv = os.path.join(args.output_dir, f"semantic_changes_{transition}.csv")
        write_combined_csv(out_csv, merged_rows, all_fields)

        counts = [len(t) for _, t in tops]
        n_common = len(common_words)
        count_str = ", ".join(f"{l}={n}" for l, n in zip(labels, counts))
        print(f"  {transition}: {count_str}, intersection={n_common}")
        summary_rows.append((transition, counts, n_common))

    # Write summary
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["transition"] + [f"{l}_top{args.topn}" for l in labels] + ["intersection"]
        writer.writerow(header)
        for transition, counts, n_common in summary_rows:
            writer.writerow([transition] + counts + [n_common])
    print(f"\nSummary written to {summary_path}")

    total_common = sum(s[2] for s in summary_rows)
    print(f"Total intersection words across all transitions: {total_common}")
    print(f"Done. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
