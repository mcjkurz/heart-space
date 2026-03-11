#!/usr/bin/env python3
"""
Find the most semantically stable words across time periods.

Identifies words that appear frequently in all periods, trains a TempRef model
with them as targets, and ranks them by semantic stability (similarity of
temporal variants).

Usage (run from project root):
  python scripts/find_stable_words.py
  python scripts/find_stable_words.py --num-targets 500 --min-freq 50
  python scripts/find_stable_words.py --epochs 5 --vector-size 300
"""

import argparse
import os

import numpy as np
from collections import Counter

from qhchina.analytics.word2vec import TempRefWord2Vec
from qhchina.utils import LineSentenceFile

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


def load_period_sentences(data_dir):
    """Load segmented sentences from per-period .txt files."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            corpora[period] = LineSentenceFile(filepath)
    return corpora


def compute_word_frequencies(period_sentences):
    """Compute word frequencies per period."""
    print("\nCounting word frequencies per period...")
    period_vocab = {}
    for period in PERIODS:
        if period not in period_sentences:
            continue
        counter = Counter()
        for sentence in period_sentences[period]:
            counter.update(sentence)
        period_vocab[period] = counter
        print(f"  {period}: {len(counter)} unique words")
    return period_vocab


def find_candidate_words(period_vocab, min_freq_per_period, min_word_length):
    """Find words that appear in all periods with minimum frequency."""
    candidates = None
    for period in PERIODS:
        if period not in period_vocab:
            continue
        words_above_threshold = set(
            word for word, count in period_vocab[period].items()
            if count >= min_freq_per_period and len(word) >= min_word_length
        )
        if candidates is None:
            candidates = words_above_threshold
        else:
            candidates = candidates.intersection(words_above_threshold)
        print(f"  After {period}: {len(candidates)} candidates")
    return candidates or set()


def calculate_stability_scores(model, target_words):
    """Calculate stability scores for each target word."""
    print("\nCalculating stability scores...")
    stability_scores = []

    for word in target_words:
        consecutive_sims = []
        for i in range(len(PERIODS) - 1):
            period1 = PERIODS[i]
            period2 = PERIODS[i + 1]
            variant1 = f"{word}_{period1}"
            variant2 = f"{word}_{period2}"
            try:
                sim = model.similarity(variant1, variant2)
                consecutive_sims.append(sim)
            except Exception:
                consecutive_sims.append(np.nan)

        first_variant = f"{word}_{PERIODS[0]}"
        last_variant = f"{word}_{PERIODS[-1]}"
        try:
            first_last_sim = model.similarity(first_variant, last_variant)
        except Exception:
            first_last_sim = np.nan

        mean_sim = np.nanmean(consecutive_sims)
        min_sim = np.nanmin(consecutive_sims)
        stability_scores.append((word, mean_sim, min_sim, consecutive_sims, first_last_sim))

    stability_scores.sort(key=lambda x: -x[1])
    return stability_scores


def print_results(stability_scores, top_n=50):
    """Print most and least stable words."""
    print(f"\n{'='*90}")
    print("MOST STABLE WORDS (highest mean consecutive similarity)")
    print(f"{'='*90}")
    print(f"{'Rank':<6}{'Word':<15}{'First-Last':<12}{'Mean Sim':<12}{'Min Sim':<12}Consecutive Similarities")
    print("-" * 90)
    for i, (word, mean_sim, min_sim, sims, first_last) in enumerate(stability_scores[:top_n]):
        sims_str = " → ".join([f"{s:.3f}" for s in sims])
        print(f"{i+1:<6}{word:<15}{first_last:<12.4f}{mean_sim:<12.4f}{min_sim:<12.4f}{sims_str}")

    print(f"\n{'='*90}")
    print("LEAST STABLE WORDS (lowest mean consecutive similarity)")
    print(f"{'='*90}")
    print(f"{'Rank':<6}{'Word':<15}{'First-Last':<12}{'Mean Sim':<12}{'Min Sim':<12}Consecutive Similarities")
    print("-" * 90)
    for i, (word, mean_sim, min_sim, sims, first_last) in enumerate(stability_scores[-top_n:]):
        sims_str = " → ".join([f"{s:.3f}" for s in sims])
        print(f"{len(stability_scores)-top_n+1+i:<6}{word:<15}{first_last:<12.4f}{mean_sim:<12.4f}{min_sim:<12.4f}{sims_str}")


def save_results(stability_scores, output_path):
    """Save stability scores to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("rank,word,mean_similarity,min_similarity,first_last_similarity,")
        f.write(",".join([f"sim_{PERIODS[i]}_to_{PERIODS[i+1]}"
                          for i in range(len(PERIODS)-1)]))
        f.write("\n")
        for i, (word, mean_sim, min_sim, sims, first_last) in enumerate(stability_scores):
            f.write(f"{i+1},{word},{mean_sim:.4f},{min_sim:.4f},{first_last:.4f},")
            f.write(",".join([f"{s:.4f}" for s in sims]))
            f.write("\n")
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find semantically stable words across time periods."
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--output", default="models/tempref_stable_words.npy",
        help="Output path for the trained model (default: models/tempref_stable_words.npy)"
    )
    parser.add_argument(
        "--results", default="results/word_stability_scores.csv",
        help="Output path for stability scores CSV (default: results/word_stability_scores.csv)"
    )
    parser.add_argument(
        "--num-targets", type=int, default=1000,
        help="Number of target words to track (default: 1000)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=100,
        help="Minimum word frequency per period (default: 100)"
    )
    parser.add_argument(
        "--min-word-length", type=int, default=2,
        help="Minimum word length (default: 2)"
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of top/bottom words to display (default: 50)"
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

    print("Loading segmented sentences...")
    period_sentences = load_period_sentences(args.data_dir)
    print(f"Loaded data for {len(period_sentences)} periods: {list(period_sentences.keys())}")

    period_vocab = compute_word_frequencies(period_sentences)

    print(f"\nFinding candidate words (min_freq={args.min_freq}, min_length={args.min_word_length})...")
    candidates = find_candidate_words(period_vocab, args.min_freq, args.min_word_length)
    print(f"Found {len(candidates)} words appearing in all periods")

    total_freq = {word: sum(period_vocab[p][word] for p in PERIODS if p in period_vocab) for word in candidates}
    sorted_candidates = sorted(total_freq.items(), key=lambda x: -x[1])
    target_words = [word for word, freq in sorted_candidates[:args.num_targets]]
    print(f"Selected top {len(target_words)} words by total frequency")
    print(f"Sample targets: {target_words[:20]}")

    if os.path.exists(args.output) and not args.overwrite:
        print(f"\nLoading existing model from {args.output}...")
        model = TempRefWord2Vec.load(args.output)
    else:
        print(f"\nTraining TempRefWord2Vec with {len(target_words)} targets (params: {params})...")
        corpora = {period: period_sentences[period] for period in PERIODS if period in period_sentences}
        model = TempRefWord2Vec(
            sentences=corpora,
            targets=target_words,
            **params,
        )
        model.train()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        model.save(args.output)
        print(f"Model saved to {args.output}")

    stability_scores = calculate_stability_scores(model, target_words)
    print_results(stability_scores, top_n=args.top_n)
    save_results(stability_scores, args.results)

    print(f"\n{'='*80}")
    print("Analysis completed")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
