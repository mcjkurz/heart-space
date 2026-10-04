#!/usr/bin/env python3
"""
Ensemble Size vs Stability Analysis.

Analyzes how stability (top-N overlap) improves as ensemble size increases.
Uses pre-trained models and performs repeated split-half comparisons for
different ensemble sizes (e.g., k=1, 3, 5, 10, 20).

Models should be trained separately using train_tempref.py and follow the
naming convention: model_e{epochs}_s{seed}.npy

Example workflow:
  # 1. Train 100 models at 3 epochs
  python scripts/train_tempref.py --trials 100 --epochs 3 --output-dir models/real/

  # 2. Run ensemble stability analysis
  python scripts/ensemble_stability.py --model-dir models/real/ --output-dir results/ensemble_stability/

Output:
  - ensemble_stability_overall.csv: Mean overlap for each ensemble size
  - ensemble_stability_by_transition.csv: Per-transition breakdown
  - ensemble_stability_raw.csv: Raw data for all splits (for plotting)
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qhchina.analytics import TempRefWord2Vec
from qhchina.helpers.texts import load_stopwords
from qhchina.utils import LineSentenceFile


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']
TRANSITIONS = [
    ('mingqing', 'late_qing'),
    ('late_qing', 'republican'),
    ('republican', 'socialist'),
    ('socialist', 'contemporary'),
]


def load_target_words(words_file: str) -> List[str]:
    with open(words_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_cooccurrences(
    data_dir: str,
    target_words: List[str],
    window_size: int = 10
) -> Dict[str, Counter]:
    cooc_count = {period: Counter() for period in PERIODS}
    target_set = set(target_words)

    print("Computing co-occurrences...")
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if not os.path.exists(filepath):
            continue
        sentences = LineSentenceFile(filepath)
        for sentence in tqdm(sentences, desc=period, leave=False):
            for word_id, word in enumerate(sentence):
                if word in target_set:
                    start = max(0, word_id - window_size)
                    end = min(len(sentence), word_id + window_size)
                    for i in range(start, end):
                        if sentence[i] != word:
                            cooc_count[period][sentence[i]] += 1
    return cooc_count


def get_semantic_change_vector(
    model: TempRefWord2Vec, 
    target: str, 
    transition: Tuple[str, str]
) -> Dict[str, float]:
    changes = model.calculate_semantic_change(target)
    from_period, to_period = transition
    transition_key = f"{from_period}_to_{to_period}"
    
    if transition_key not in changes:
        raise KeyError(f"Transition {transition_key} not found in model")
    
    return {word: score for word, score in changes[transition_key]}


def filter_delta_x(
    delta_x: Dict[str, float],
    model: TempRefWord2Vec,
    transition: Tuple[str, str],
    cooc_count: Dict[str, Counter],
    stopwords: Set[str],
    min_word_length: int = 2,
    min_cooc: int = 5,
    min_count: int = 5,
) -> Dict[str, float]:
    from_period, to_period = transition
    from_counts = model.get_period_vocab_counts(from_period)
    to_counts = model.get_period_vocab_counts(to_period)
    
    filtered = {}
    for word, score in delta_x.items():
        if word in stopwords:
            continue
        if len(word) < min_word_length:
            continue
        if from_counts[word] < min_count or to_counts[word] < min_count:
            continue
        if cooc_count[to_period][word] < min_cooc:
            continue
        if score <= 0:
            continue
        filtered[word] = score
    
    return filtered


def compute_mean_delta_x(delta_x_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not delta_x_list:
        return {}
    
    all_words = set()
    for dx in delta_x_list:
        all_words.update(dx.keys())
    
    mean_delta_x = {}
    for word in all_words:
        values = [dx.get(word, 0.0) for dx in delta_x_list]
        mean_delta_x[word] = np.mean(values)
    
    return mean_delta_x


def get_top_n_words(delta_x: Dict[str, float], n: int) -> List[str]:
    sorted_words = sorted(delta_x.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:n]]


def compute_top_n_overlap(
    delta_x_1: Dict[str, float],
    delta_x_2: Dict[str, float],
    top_n: int
) -> float:
    top_words_1 = set(get_top_n_words(delta_x_1, top_n))
    top_words_2 = set(get_top_n_words(delta_x_2, top_n))
    intersection = top_words_1 & top_words_2
    return len(intersection) / top_n


def discover_models(model_dir: str, epochs: int = None) -> Dict[int, List[Tuple[int, str]]]:
    """
    Scan model_dir for files matching model_e{epochs}_s{seed}.npy.
    Returns {epochs: [(seed, filepath), ...]}.
    If epochs is specified, only return models with that epoch count.
    """
    pattern = re.compile(r"^model_e(\d+)_s(\d+)\.npy$")
    epoch_models: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            e = int(m.group(1))
            seed = int(m.group(2))
            if epochs is None or e == epochs:
                epoch_models[e].append((seed, os.path.join(model_dir, fname)))

    for e in epoch_models:
        epoch_models[e].sort(key=lambda x: x[0])

    return dict(epoch_models)


def run_ensemble_comparison(
    all_delta_x: Dict[Tuple[str, str], List[Dict[str, float]]],
    ensemble_sizes: List[int],
    n_splits: int,
    top_n: int,
    rng_seed: int = 42
) -> Dict[int, Dict[Tuple[str, str], Dict[str, List[float]]]]:
    """
    Run ensemble size comparison with repeated random splits.
    
    For each ensemble size k:
      - Repeatedly sample two disjoint groups of size k
      - Compute mean Δx for each group
      - Compute top-N overlap
    """
    import random
    rng = random.Random(rng_seed)
    
    results = {k: {t: {"overlap": []} for t in all_delta_x.keys()} for k in ensemble_sizes}
    
    for transition, delta_x_list in all_delta_x.items():
        n_total = len(delta_x_list)
        indices = list(range(n_total))
        
        for k in ensemble_sizes:
            if n_total < 2 * k:
                print(f"  WARNING: {transition} has {n_total} models, need {2*k} for k={k}")
                continue
            
            for _ in range(n_splits):
                rng.shuffle(indices)
                group_a_idx = indices[:k]
                group_b_idx = indices[k:2*k]
                
                group_a = [delta_x_list[i] for i in group_a_idx]
                group_b = [delta_x_list[i] for i in group_b_idx]
                
                mean_a = compute_mean_delta_x(group_a)
                mean_b = compute_mean_delta_x(group_b)
                
                overlap = compute_top_n_overlap(mean_a, mean_b, top_n)
                results[k][transition]["overlap"].append(overlap)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how ensemble size affects stability of semantic change results."
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing trained models (model_e{epochs}_s{seed}.npy)"
    )
    parser.add_argument(
        "--output-dir", default="results/ensemble_stability",
        help="Output directory for results (default: results/ensemble_stability)"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Replacement token for target words"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Filter to models with this epoch count (default: auto-detect most common)"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Top-N words to analyze for stability (default: 100)"
    )
    parser.add_argument(
        "--ensemble-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20],
        help="Ensemble sizes to compare (default: 1 3 5 10 20)"
    )
    parser.add_argument(
        "--n-splits", type=int, default=200,
        help="Number of random splits per ensemble size (default: 200)"
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for split sampling (default: 42)"
    )
    args = parser.parse_args()

    # Validate ensemble sizes
    max_ensemble = max(args.ensemble_sizes)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover models
    print(f"Scanning {args.model_dir} for models...")
    discovered = discover_models(args.model_dir, args.epochs)
    
    if not discovered:
        print(f"ERROR: No models found in {args.model_dir}")
        print(f"  Expected files matching: model_e{{epochs}}_s{{seed}}.npy")
        return
    
    # If epochs not specified, use the most common
    if args.epochs is None:
        args.epochs = max(discovered.keys(), key=lambda e: len(discovered[e]))
    
    model_entries = discovered.get(args.epochs, [])
    n_models = len(model_entries)
    
    print(f"  Found {n_models} models at {args.epochs} epochs")
    
    if n_models < 2 * max_ensemble:
        print(f"  ERROR: Need at least {2 * max_ensemble} models for ensemble size {max_ensemble}")
        print(f"         Only have {n_models}. Train more models or reduce --ensemble-sizes.")
        return
    
    # Load target words and compute co-occurrences
    print(f"\nLoading target words from {args.words}...")
    target_words = load_target_words(args.words)
    print(f"  Found {len(target_words)} target words")
    
    cooc_count = compute_cooccurrences(args.data_dir, target_words)
    stopwords = load_stopwords("zh_sim")
    print(f"  Loaded {len(stopwords)} stopwords")
    
    # Load all models and extract delta_x
    print(f"\n{'='*70}")
    print("LOADING MODELS AND EXTRACTING SEMANTIC CHANGES")
    print(f"{'='*70}")
    
    all_delta_x = {t: [] for t in TRANSITIONS}
    loaded_seeds = []
    
    for seed, model_path in tqdm(model_entries, desc="Loading"):
        model = TempRefWord2Vec.load(model_path)
        loaded_seeds.append(seed)
        
        for transition in TRANSITIONS:
            try:
                delta_x = get_semantic_change_vector(model, args.target, transition)
                delta_x_filtered = filter_delta_x(
                    delta_x, model, transition, cooc_count, stopwords
                )
                all_delta_x[transition].append(delta_x_filtered)
            except KeyError:
                all_delta_x[transition].append({})
    
    print(f"  Loaded {len(loaded_seeds)} models")
    
    # Run ensemble comparison
    print(f"\n{'='*70}")
    print(f"RUNNING ENSEMBLE SIZE COMPARISON")
    print(f"{'='*70}")
    print(f"  Ensemble sizes: {args.ensemble_sizes}")
    print(f"  Splits per size: {args.n_splits}")
    print(f"  Top-N: {args.top_n}")
    
    results = run_ensemble_comparison(
        all_delta_x,
        ensemble_sizes=args.ensemble_sizes,
        n_splits=args.n_splits,
        top_n=args.top_n,
        rng_seed=args.split_seed
    )
    
    # Report results
    print(f"\n{'='*70}")
    print("ENSEMBLE SIZE VS STABILITY")
    print(f"{'='*70}")
    
    summary_data = []
    
    for transition in TRANSITIONS:
        transition_key = f"{transition[0]}_to_{transition[1]}"
        print(f"\n{transition_key}")
        print("-" * 70)
        print(f"  {'Ensemble':<10} {'Mean overlap':>14} {'Std':>10} {'95% CI':<20}")
        print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*20}")
        
        for k in args.ensemble_sizes:
            overlaps = results[k][transition]["overlap"]
            if not overlaps:
                print(f"  {k:<10} {'N/A':>14}")
                continue
            
            overlaps = np.array(overlaps)
            mean_ov = np.mean(overlaps)
            std_ov = np.std(overlaps)
            ci_low, ci_high = np.percentile(overlaps, [2.5, 97.5])
            
            print(f"  {k:<10} {mean_ov*100:>13.1f}% {std_ov*100:>9.1f}% [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
            
            summary_data.append({
                "transition": transition_key,
                "ensemble_size": k,
                "n_splits": len(overlaps),
                "overlap_mean": mean_ov,
                "overlap_std": std_ov,
                "overlap_ci_low": ci_low,
                "overlap_ci_high": ci_high,
            })
    
    # Overall average
    print(f"\n{'='*70}")
    print("OVERALL AVERAGE (across all transitions)")
    print(f"{'='*70}")
    print(f"\n  {'Ensemble':<10} {'Mean overlap':>14} {'Std':>10} {'95% CI':<20}")
    print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*20}")
    
    overall_data = []
    
    for k in args.ensemble_sizes:
        all_overlaps = []
        for transition in TRANSITIONS:
            all_overlaps.extend(results[k][transition]["overlap"])
        
        if not all_overlaps:
            continue
        
        all_overlaps = np.array(all_overlaps)
        mean_ov = np.mean(all_overlaps)
        std_ov = np.std(all_overlaps)
        ci_low, ci_high = np.percentile(all_overlaps, [2.5, 97.5])
        
        print(f"  {k:<10} {mean_ov*100:>13.1f}% {std_ov*100:>9.1f}% [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
        
        overall_data.append({
            "ensemble_size": k,
            "overlap_mean": mean_ov,
            "overlap_std": std_ov,
            "overlap_ci_low": ci_low,
            "overlap_ci_high": ci_high,
        })
    
    # Save results
    results_path = os.path.join(args.output_dir, "ensemble_stability_by_transition.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["transition", "ensemble_size", "n_splits", 
                      "overlap_mean", "overlap_std", "overlap_ci_low", "overlap_ci_high"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\n\nPer-transition results saved to {results_path}")
    
    overall_path = os.path.join(args.output_dir, "ensemble_stability_overall.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["ensemble_size", "overlap_mean", "overlap_std", 
                      "overlap_ci_low", "overlap_ci_high"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall_data)
    print(f"Overall results saved to {overall_path}")
    
    # Save raw data
    raw_path = os.path.join(args.output_dir, "ensemble_stability_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["transition", "ensemble_size", "split_id", "overlap"])
        for k in args.ensemble_sizes:
            for transition in TRANSITIONS:
                transition_key = f"{transition[0]}_to_{transition[1]}"
                for i, ov in enumerate(results[k][transition]["overlap"]):
                    writer.writerow([transition_key, k, i+1, ov])
    print(f"Raw data saved to {raw_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if overall_data:
        first = overall_data[0]
        last = overall_data[-1]
        
        print(f"\n  Single model (k=1):     {first['overlap_mean']*100:.0f}% overlap")
        print(f"  Largest ensemble (k={last['ensemble_size']}): {last['overlap_mean']*100:.0f}% overlap")
        print(f"\n  Improvement: +{(last['overlap_mean'] - first['overlap_mean'])*100:.0f} percentage points")


if __name__ == "__main__":
    main()
