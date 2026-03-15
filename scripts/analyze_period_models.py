#!/usr/bin/env python3
"""
Analyze trained per-period Word2Vec models to find words most similar to a
target, averaging similarity scores across multiple models (seeds) for stability.

Requires multiple models per period - single-model analysis is not supported
since individual models are unstable.

Usage (run from project root):
  # Analyze per-period models
  python scripts/analyze_period_models.py --model-dir models/period_models/

  # With frequency filtering
  python scripts/analyze_period_models.py --model-dir models/period_models/ --min-freq 10 --top-n 50
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from qhchina.analytics.vectors import cosine_similarity
from qhchina.analytics.word2vec import Word2Vec
from qhchina.utils import LineSentenceFile


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']


def compute_word_frequencies(data_dir: str) -> Dict[str, Counter]:
    """Compute word frequencies per period from sentence files."""
    print("Computing word frequencies...")
    freqs = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            freqs[period] = Counter()
            for sentence in tqdm(LineSentenceFile(filepath), desc=f"  {period}", leave=False):
                freqs[period].update(sentence)
    return freqs


def discover_period_models(model_dir: str) -> Dict[str, List[str]]:
    """
    Discover per-period model files in a directory.

    Expects filenames like: {period}_e{epochs}_s{seed}.npy
    Returns: {period: [path1, path2, ...]} sorted by path.
    """
    if not os.path.exists(model_dir):
        return {}
    
    period_models: Dict[str, List[str]] = defaultdict(list)
    filename_re = re.compile(r"^(.+)_e(\d+)_s(\d+)\.npy$")

    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".npy"):
            continue
        m = filename_re.match(fname)
        if m:
            period = m.group(1)
            if period in PERIODS:
                period_models[period].append(os.path.join(model_dir, fname))

    return dict(period_models)


def collect_similarities_from_models(
    model_paths: List[str],
    target_word: str,
) -> Dict[str, List[float]]:
    """
    Load each Word2Vec model and compute cosine similarity of target_word
    against the entire vocabulary.

    Returns: {word: [sim_model1, sim_model2, ...]}
    """
    word_scores: Dict[str, List[float]] = defaultdict(list)

    for model_path in tqdm(model_paths, desc="  Loading models", leave=False):
        model = Word2Vec.load(model_path)
        if target_word not in model:
            continue

        target_idx = model.vocab[target_word]
        target_vec = model.W[target_idx].reshape(1, -1)
        all_sims = cosine_similarity(target_vec, model.W).flatten()

        for idx, sim in enumerate(all_sims):
            if idx == target_idx:
                continue
            word = model.index2word[idx]
            word_scores[word].append(float(sim))

    return dict(word_scores)


def aggregate_and_rank(
    word_scores: Dict[str, List[float]],
    period_freqs: Optional[Counter],
    min_freq: int,
    top_n: int,
) -> List[dict]:
    """
    Aggregate similarity scores across models and rank by mean similarity.

    Returns list of row dicts sorted by mean_similarity descending.
    """
    rows = []
    for word, scores in word_scores.items():
        # Frequency filter
        if period_freqs is not None and period_freqs.get(word, 0) < min_freq:
            continue

        scores_arr = np.array(scores)
        mean_sim = np.mean(scores_arr)
        std_sim = np.std(scores_arr, ddof=1) if len(scores_arr) > 1 else 0.0
        ci_low = np.percentile(scores_arr, 2.5) if len(scores_arr) > 1 else mean_sim
        ci_high = np.percentile(scores_arr, 97.5) if len(scores_arr) > 1 else mean_sim

        rows.append({
            "word": word,
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_models": len(scores_arr),
        })

    rows.sort(key=lambda r: r["mean_similarity"], reverse=True)
    return rows[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-period Word2Vec models to find words most similar "
                    "to a target, averaging across multiple seeds for stability."
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing per-period Word2Vec models ({period}_e{epochs}_s{seed}.npy)"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target word to query similarities for (default: interiority)"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for CSV results (default: results)"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of most similar words to report per period (default: 100)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=10,
        help="Minimum word frequency in the respective period to include (default: 10)"
    )

    args = parser.parse_args()

    print(f"Scanning {args.model_dir} for models...")
    period_models = discover_period_models(args.model_dir)
    
    if not period_models:
        print(f"ERROR: No period model files found in {args.model_dir}")
        print(f"  Expected files matching pattern: {{period}}_e{{epochs}}_s{{seed}}.npy")
        return

    # Check minimum models
    for period in PERIODS:
        if period in period_models:
            n = len(period_models[period])
            if n < 2:
                print(f"  WARNING: {period} has only {n} model(s). Need ≥2 for stable averaging.")
            else:
                print(f"  {period}: {n} models")
        else:
            print(f"  {period}: no models found")

    period_freqs = compute_word_frequencies(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    all_period_rows = {}
    for period in PERIODS:
        if period not in period_models:
            continue

        model_paths = period_models[period]
        n_models = len(model_paths)
        print(f"\nAnalyzing {period} ({n_models} models)...")

        word_scores = collect_similarities_from_models(model_paths, args.target)

        if not word_scores:
            print(f"  '{args.target}' not found in any {period} model")
            continue

        rows = aggregate_and_rank(
            word_scores,
            period_freqs=period_freqs.get(period),
            min_freq=args.min_freq,
            top_n=args.top_n,
        )

        for row in rows:
            for p in PERIODS:
                row[f"freq_{p}"] = period_freqs.get(p, Counter()).get(row["word"], 0)

        all_period_rows[period] = rows

        print(f"  Top {min(len(rows), 20)} similar to '{args.target}':")
        for i, row in enumerate(rows[:20], 1):
            freq_str = ", ".join(f"{p[:3]}:{row.get(f'freq_{p}', 0)}" for p in PERIODS)
            print(f"    {i:2}. {row['word']}: {row['mean_similarity']:.4f} "
                  f"(±{row['std_similarity']:.4f}, n={row['n_models']}) ({freq_str})")

    # Write CSV
    csv_path = os.path.join(args.output_dir, "period_model_similar_words.csv")
    fieldnames = [
        "period", "rank", "word", "mean_similarity", "std_similarity",
        "ci_low", "ci_high", "n_models",
    ] + [f"freq_{p}" for p in PERIODS]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for period in PERIODS:
            for i, row in enumerate(all_period_rows.get(period, []), 1):
                out_row = {"period": period, "rank": i}
                for key in fieldnames:
                    if key in row:
                        val = row[key]
                        if isinstance(val, float):
                            out_row[key] = f"{val:.6f}"
                        else:
                            out_row[key] = val
                    elif key not in out_row:
                        out_row[key] = ""
                writer.writerow(out_row)

    print(f"\nResults saved to {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
