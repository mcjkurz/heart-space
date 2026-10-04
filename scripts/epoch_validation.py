#!/usr/bin/env python3
"""
Epoch Validation: Compare models trained at different epoch counts.

Analyzes within-epoch stability and cross-epoch convergence to determine
the optimal training length. Uses pre-trained models and produces:

  1. Within-epoch stability – pairwise seed-to-seed agreement at each epoch
     setting (Spearman ρ, top-N overlap).
  2. Cross-epoch convergence – do the *mean* results of one epoch setting agree
     with those of another?
  3. Progressive overlap – how much do the top-N lists change as epochs increase?

Models should be trained separately using train_tempref.py and follow the
naming convention: model_e{epochs}_s{seed}.npy

Example workflow:
  # 1. Train models at different epoch counts
  python scripts/train_tempref.py --trials 10 --epochs 1 --seed-start 1 --output-dir models/epochs/
  python scripts/train_tempref.py --trials 10 --epochs 3 --seed-start 1 --output-dir models/epochs/
  python scripts/train_tempref.py --trials 10 --epochs 5 --seed-start 1 --output-dir models/epochs/
  python scripts/train_tempref.py --trials 10 --epochs 7 --seed-start 1 --output-dir models/epochs/

  # 2. Run epoch validation analysis
  python scripts/epoch_validation.py --model-dir models/epochs/ --output-dir results/epoch_validation/

Output:
  - within_epoch_stability.csv: Pairwise overlap for each (epoch, transition)
  - epoch_comparison_results.csv: Cross-epoch mean comparisons
  - progressive_jaccard_overlap.csv: Overlap matrix between epoch settings
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.stats import spearmanr
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

    all_words: Set[str] = set()
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


def compute_spearman_correlation(
    delta_x_1: Dict[str, float],
    delta_x_2: Dict[str, float]
) -> Tuple[float, int]:
    common_words = set(delta_x_1.keys()) & set(delta_x_2.keys())

    if len(common_words) < 10:
        return np.nan, len(common_words)

    words = sorted(common_words)
    vec1 = [delta_x_1[w] for w in words]
    vec2 = [delta_x_2[w] for w in words]

    corr, _ = spearmanr(vec1, vec2)
    return corr, len(common_words)


def compute_top_n_stability(
    delta_x_1: Dict[str, float],
    delta_x_2: Dict[str, float],
    top_n: int
) -> Dict[str, float]:
    top_words_1 = get_top_n_words(delta_x_1, top_n)
    top_words_2 = get_top_n_words(delta_x_2, top_n)

    set_1 = set(top_words_1)
    set_2 = set(top_words_2)

    intersection = set_1 & set_2
    overlap = len(intersection) / top_n

    if len(intersection) < 5:
        return {"overlap": overlap, "rank_corr": np.nan}

    rank_in_top_1 = {w: i + 1 for i, w in enumerate(top_words_1)}
    rank_in_top_2 = {w: i + 1 for i, w in enumerate(top_words_2)}

    common_words = sorted(intersection)
    ranks_1 = [rank_in_top_1[w] for w in common_words]
    ranks_2 = [rank_in_top_2[w] for w in common_words]

    rank_corr, _ = spearmanr(ranks_1, ranks_2)
    return {"overlap": overlap, "rank_corr": rank_corr}


def compute_within_epoch_stability(
    delta_x_list: List[Dict[str, float]],
    top_n: int
) -> Dict[str, float]:
    if len(delta_x_list) < 2:
        return {k: np.nan for k in [
            "mean_pairwise_spearman", "std_pairwise_spearman",
            "mean_top_n_overlap", "std_top_n_overlap",
            "mean_top_n_rank_corr", "std_top_n_rank_corr",
        ]}

    pairwise_spearman = []
    pairwise_overlap = []
    pairwise_rank_corr = []

    for i, j in combinations(range(len(delta_x_list)), 2):
        corr, _ = compute_spearman_correlation(delta_x_list[i], delta_x_list[j])
        if not np.isnan(corr):
            pairwise_spearman.append(corr)

        top_n_metrics = compute_top_n_stability(
            delta_x_list[i], delta_x_list[j], top_n
        )
        pairwise_overlap.append(top_n_metrics["overlap"])
        if not np.isnan(top_n_metrics["rank_corr"]):
            pairwise_rank_corr.append(top_n_metrics["rank_corr"])

    return {
        "mean_pairwise_spearman": np.mean(pairwise_spearman) if pairwise_spearman else np.nan,
        "std_pairwise_spearman": np.std(pairwise_spearman) if pairwise_spearman else np.nan,
        "mean_top_n_overlap": np.mean(pairwise_overlap) if pairwise_overlap else np.nan,
        "std_top_n_overlap": np.std(pairwise_overlap) if pairwise_overlap else np.nan,
        "mean_top_n_rank_corr": np.mean(pairwise_rank_corr) if pairwise_rank_corr else np.nan,
        "std_top_n_rank_corr": np.std(pairwise_rank_corr) if pairwise_rank_corr else np.nan,
    }


def discover_models(model_dir: str) -> Dict[int, List[Tuple[int, str]]]:
    """
    Scan model_dir for files matching model_e{epochs}_s{seed}.npy.
    Returns {epochs: [(seed, filepath), ...]}.
    """
    pattern = re.compile(r"^model_e(\d+)_s(\d+)\.npy$")
    epoch_models: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    for fname in os.listdir(model_dir):
        m = pattern.match(fname)
        if m:
            epochs = int(m.group(1))
            seed = int(m.group(2))
            epoch_models[epochs].append((seed, os.path.join(model_dir, fname)))

    for epochs in epoch_models:
        epoch_models[epochs].sort(key=lambda x: x[0])

    return dict(epoch_models)


def _save_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze within-epoch stability and cross-epoch convergence."
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing trained models (model_e{epochs}_s{seed}.npy)"
    )
    parser.add_argument(
        "--output-dir", default="results/epoch_validation",
        help="Output directory for results (default: results/epoch_validation)"
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
        "--epochs", nargs="+", type=int, default=None,
        help="Epoch values to analyze (default: auto-discover from model files)"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Top-N words to analyze for stability (default: 100)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover models
    print(f"Scanning {args.model_dir} for models...")
    discovered = discover_models(args.model_dir)
    
    if not discovered:
        print(f"ERROR: No models found in {args.model_dir}")
        print(f"  Expected files matching: model_e{{epochs}}_s{{seed}}.npy")
        return

    epoch_values = sorted(args.epochs if args.epochs else discovered.keys())
    
    print(f"\n{'='*70}")
    print("DISCOVERED MODELS")
    print(f"{'='*70}")
    for e in epoch_values:
        seeds_for_e = discovered.get(e, [])
        seed_list = [s for s, _ in seeds_for_e]
        print(f"  epochs={e}: {len(seeds_for_e)} models  (seeds: {seed_list})")

    # Load target words and compute co-occurrences
    print(f"\nLoading target words from {args.words}...")
    target_words = load_target_words(args.words)
    print(f"  Found {len(target_words)} target words")

    cooc_count = compute_cooccurrences(args.data_dir, target_words)
    stopwords = load_stopwords("zh_sim")
    print(f"  Loaded {len(stopwords)} stopwords")

    # Load models and extract delta_x
    epoch_results: Dict[int, Dict[Tuple, List[Dict[str, float]]]] = {}

    for epochs in epoch_values:
        epoch_results[epochs] = {t: [] for t in TRANSITIONS}
        model_entries = discovered.get(epochs, [])
        
        for seed, model_path in tqdm(model_entries, desc=f"Loading e{epochs}"):
            model = TempRefWord2Vec.load(model_path)
            for transition in TRANSITIONS:
                try:
                    delta_x = get_semantic_change_vector(model, args.target, transition)
                    delta_x_filtered = filter_delta_x(
                        delta_x, model, transition, cooc_count, stopwords
                    )
                    epoch_results[epochs][transition].append(delta_x_filtered)
                except KeyError as e_err:
                    print(f"    e{epochs} s{seed} {transition}: SKIPPED ({e_err})")

    sorted_epochs = sorted(epoch_results.keys())

    # ==================================================================
    # 1. WITHIN-EPOCH STABILITY
    # ==================================================================
    print(f"\n{'='*70}")
    print("1. WITHIN-EPOCH STABILITY (pairwise seed-to-seed agreement)")
    print(f"{'='*70}")

    within_epoch_rows = []

    for transition in TRANSITIONS:
        tkey = f"{transition[0]}_to_{transition[1]}"
        print(f"\n  {tkey}")
        print(f"  {'Epochs':<8} {'N seeds':>8} {'Mean ρ':>10} {'Std ρ':>10} "
              f"{'Top-{0} overlap':>16} {'Std':>8}".format(args.top_n))
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*16} {'-'*8}")

        for epochs in sorted_epochs:
            dx_list = epoch_results[epochs][transition]
            n_seeds = len(dx_list)
            stab = compute_within_epoch_stability(dx_list, args.top_n)

            def _fmt(v, pct=False):
                if np.isnan(v):
                    return "N/A"
                return f"{v*100:.1f}%" if pct else f"{v:.4f}"

            print(f"  {epochs:<8} {n_seeds:>8} "
                  f"{_fmt(stab['mean_pairwise_spearman']):>10} "
                  f"{_fmt(stab['std_pairwise_spearman']):>10} "
                  f"{_fmt(stab['mean_top_n_overlap'], pct=True):>16} "
                  f"{_fmt(stab['std_top_n_overlap'], pct=True):>8}")

            within_epoch_rows.append({
                "transition": tkey,
                "epochs": epochs,
                "n_seeds": n_seeds,
                "mean_pairwise_spearman": stab["mean_pairwise_spearman"],
                "std_pairwise_spearman": stab["std_pairwise_spearman"],
                "mean_top_n_overlap": stab["mean_top_n_overlap"],
                "std_top_n_overlap": stab["std_top_n_overlap"],
                "mean_top_n_rank_corr": stab["mean_top_n_rank_corr"],
                "std_top_n_rank_corr": stab["std_top_n_rank_corr"],
                "top_n": args.top_n,
            })

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  SUMMARY (averaged across all transitions)")
    print(f"  {'Epochs':<8} {'Mean ρ':>12} {'Top-{} overlap':>18}".format(args.top_n))
    print(f"  {'-'*8} {'-'*12} {'-'*18}")

    for epochs in sorted_epochs:
        rows = [r for r in within_epoch_rows if r["epochs"] == epochs]
        if rows:
            avg_rho = np.nanmean([r["mean_pairwise_spearman"] for r in rows])
            avg_overlap = np.nanmean([r["mean_top_n_overlap"] for r in rows])
            print(f"  {epochs:<8} {avg_rho:>12.4f} {avg_overlap*100:>17.1f}%")

    within_path = os.path.join(args.output_dir, "within_epoch_stability.csv")
    _save_csv(within_path, within_epoch_rows, [
        "transition", "epochs", "n_seeds",
        "mean_pairwise_spearman", "std_pairwise_spearman",
        "mean_top_n_overlap", "std_top_n_overlap",
        "mean_top_n_rank_corr", "std_top_n_rank_corr", "top_n",
    ])
    print(f"\n  → Saved to {within_path}")

    # ==================================================================
    # 2. CROSS-EPOCH CONVERGENCE
    # ==================================================================
    print(f"\n{'='*70}")
    print("2. CROSS-EPOCH CONVERGENCE (comparing mean Δx across epoch settings)")
    print(f"{'='*70}")

    epoch_means: Dict[int, Dict[Tuple, Dict[str, float]]] = {
        e: {} for e in sorted_epochs
    }
    for epochs in sorted_epochs:
        for transition in TRANSITIONS:
            dx_list = epoch_results[epochs][transition]
            if dx_list:
                epoch_means[epochs][transition] = compute_mean_delta_x(dx_list)

    cross_epoch_rows = []

    for transition in TRANSITIONS:
        tkey = f"{transition[0]}_to_{transition[1]}"
        print(f"\n  {tkey}")
        print(f"  {'Comparison':<15} {'Global ρ':>10} {'N words':>10} "
              f"{'Top-{} overlap':>16}".format(args.top_n))
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*16}")

        for e1, e2 in combinations(sorted_epochs, 2):
            if transition not in epoch_means[e1] or transition not in epoch_means[e2]:
                continue

            mean_1 = epoch_means[e1][transition]
            mean_2 = epoch_means[e2][transition]

            global_corr, n_words = compute_spearman_correlation(mean_1, mean_2)
            top_n_metrics = compute_top_n_stability(mean_1, mean_2, args.top_n)

            overlap_pct = f"{top_n_metrics['overlap']*100:.1f}%"

            print(f"  e{e1} vs e{e2:<8} {global_corr:>10.4f} {n_words:>10} "
                  f"{overlap_pct:>16}")

            cross_epoch_rows.append({
                "transition": tkey,
                "epoch_1": e1,
                "epoch_2": e2,
                "global_spearman": global_corr,
                "n_common_words": n_words,
                "top_n": args.top_n,
                "top_n_overlap": top_n_metrics["overlap"],
                "top_n_rank_corr": top_n_metrics["rank_corr"],
            })

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  SUMMARY (averaged across all transitions)")
    print(f"  {'Comparison':<15} {'Global ρ':>12} {'Top-{} overlap':>18}".format(args.top_n))
    print(f"  {'-'*15} {'-'*12} {'-'*18}")

    for e1, e2 in combinations(sorted_epochs, 2):
        rows = [r for r in cross_epoch_rows
                if r["epoch_1"] == e1 and r["epoch_2"] == e2]
        if rows:
            avg_rho = np.nanmean([r["global_spearman"] for r in rows])
            avg_overlap = np.nanmean([r["top_n_overlap"] for r in rows])
            print(f"  e{e1} vs e{e2:<8} {avg_rho:>12.4f} {avg_overlap*100:>17.1f}%")

    cross_path = os.path.join(args.output_dir, "epoch_comparison_results.csv")
    _save_csv(cross_path, cross_epoch_rows, [
        "transition", "epoch_1", "epoch_2", "global_spearman",
        "n_common_words", "top_n", "top_n_overlap", "top_n_rank_corr",
    ])
    print(f"\n  → Saved to {cross_path}")

    # ==================================================================
    # 3. PROGRESSIVE OVERLAP
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"3. PROGRESSIVE TOP-{args.top_n} OVERLAP (epoch → epoch)")
    print(f"{'='*70}")

    jaccard_rows = []

    for transition in TRANSITIONS:
        tkey = f"{transition[0]}_to_{transition[1]}"
        top_words_per_epoch = {}
        for epochs in sorted_epochs:
            if transition in epoch_means[epochs]:
                top_words_per_epoch[epochs] = set(
                    get_top_n_words(epoch_means[epochs][transition], args.top_n)
                )

        if len(top_words_per_epoch) < 2:
            continue

        print(f"\n  {tkey}")
        header = f"  {'':>8}"
        for e2 in sorted_epochs:
            header += f"  {'e'+str(e2):>8}"
        print(header)

        for e1 in sorted_epochs:
            row_str = f"  {'e'+str(e1):>8}"
            for e2 in sorted_epochs:
                if e1 in top_words_per_epoch and e2 in top_words_per_epoch:
                    s1, s2 = top_words_per_epoch[e1], top_words_per_epoch[e2]
                    jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
                    overlap_frac = len(s1 & s2) / args.top_n
                    row_str += f"  {overlap_frac*100:>7.1f}%"
                    if e1 < e2:
                        jaccard_rows.append({
                            "transition": tkey,
                            "epoch_1": e1,
                            "epoch_2": e2,
                            "top_n": args.top_n,
                            "intersection_size": len(s1 & s2),
                            "union_size": len(s1 | s2),
                            "jaccard": jaccard,
                            "overlap_fraction": overlap_frac,
                        })
                else:
                    row_str += f"  {'N/A':>8}"
            print(row_str)

    jaccard_path = os.path.join(args.output_dir, "progressive_jaccard_overlap.csv")
    _save_csv(jaccard_path, jaccard_rows, [
        "transition", "epoch_1", "epoch_2", "top_n",
        "intersection_size", "union_size", "jaccard", "overlap_fraction",
    ])
    print(f"\n  → Saved to {jaccard_path}")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
