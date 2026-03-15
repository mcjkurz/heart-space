#!/usr/bin/env python3
"""
Semantic change analysis using trained TempRefWord2Vec model(s).

Supports two modes:
  1. Single-model mode (--model): load one model, compute and filter semantic changes.
  2. Multi-model ensemble mode (--model-dir): load multiple models, aggregate scores
     with mean/std/CI/z-score, and optionally compute p-values from null models.

Usage (run from project root):
  # Single model analysis
  python scripts/semantic_change.py --model models/real/model_e3_s42.npy
  python scripts/semantic_change.py --model models/real/model_e3_s42.npy --postag "n.*" --no-plots

  # Multi-model ensemble analysis
  python scripts/semantic_change.py --model-dir models/real/ --output-dir results/multiseed/

  # Multi-model with null distribution for p-values
  python scripts/semantic_change.py --model-dir models/real/ --null-dir models/null/ --output-dir results/multiseed/

  # Null distribution analysis
  python scripts/semantic_change.py --model-dir models/null/ --output-dir results/null_distribution/
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import jieba.posseg as pseg
import numpy as np
from tqdm.auto import tqdm

from qhchina.analytics import TempRefWord2Vec
from qhchina.helpers.texts import load_stopwords
from qhchina.utils import LineSentenceFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def load_period_sentences(data_dir: str) -> Dict[str, LineSentenceFile]:
    """Load segmented sentences from per-period .txt files."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            corpora[period] = LineSentenceFile(filepath)
    return corpora


def compute_cooccurrences(
    period_sentences_dict: Dict[str, LineSentenceFile],
    target_words: List[str],
    window_size: int = 10
) -> Dict[str, Counter]:
    """Compute co-occurrence counts for target words within a window."""
    cooc_count = {period: Counter() for period in period_sentences_dict.keys()}
    target_set = set(target_words)

    print("Computing co-occurrences...")
    for period, sentences in period_sentences_dict.items():
        for sentence in tqdm(sentences, desc=period, leave=False):
            for word_id, word in enumerate(sentence):
                if word in target_set:
                    start = max(0, word_id - window_size)
                    end = min(len(sentence), word_id + window_size)
                    for i in range(start, end):
                        if sentence[i] != word:
                            cooc_count[period][sentence[i]] += 1
    return cooc_count


def compute_word_counts(data_dir: str) -> Dict[str, Counter]:
    """Compute word counts per period from sentence files."""
    word_counts = {period: Counter() for period in PERIODS}

    print("Computing word counts...")
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if not os.path.exists(filepath):
            continue
        sentences = LineSentenceFile(filepath)
        for sentence in tqdm(sentences, desc=period, leave=False):
            word_counts[period].update(sentence)

    return word_counts


# ---------------------------------------------------------------------------
# Single-model mode (original behavior)
# ---------------------------------------------------------------------------

def filter_semantic_changes(
    changes: Dict[str, List[Tuple[str, float]]],
    model: TempRefWord2Vec,
    top_vocab: Optional[int] = None,
    min_word_length: Optional[int] = None,
    min_cooc: Optional[int] = None,
    cooc_count: Optional[Dict[str, Counter]] = None,
    min_count: Optional[Tuple[int, int]] = None,
    postag: Optional[str] = None,
    min_change: Optional[float] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """Filter semantic change results based on vocabulary frequency, word length, and co-occurrence."""
    filtered_changes = {}

    for transition, word_changes in changes.items():
        filtered_words = word_changes.copy()

        if min_word_length is not None:
            filtered_words = [c for c in filtered_words if len(c[0]) >= min_word_length]

        if top_vocab is not None:
            periods = transition.split("_to_")
            if len(periods) == 2:
                from_period, to_period = periods
                from_counts = model.get_period_vocab_counts(from_period)
                to_counts = model.get_period_vocab_counts(to_period)
                top_words = (
                    set(w for w, _ in from_counts.most_common(top_vocab))
                    | set(w for w, _ in to_counts.most_common(top_vocab))
                )
                filtered_words = [c for c in filtered_words if c[0] in top_words]

        if min_count is not None:
            periods = transition.split("_to_")
            if len(periods) == 2:
                from_period, to_period = periods
                min_from, min_to = min_count
                from_counts = model.get_period_vocab_counts(from_period)
                to_counts = model.get_period_vocab_counts(to_period)
                filtered_words = [
                    c for c in filtered_words
                    if from_counts[c[0]] >= min_from and to_counts[c[0]] >= min_to
                ]

        if min_cooc is not None and cooc_count is not None:
            periods = transition.split("_to_")
            if len(periods) == 2:
                to_period = periods[1]
                filtered_words = [
                    c for c in filtered_words
                    if cooc_count[to_period][c[0]] >= min_cooc
                ]

        if postag is not None:
            filtered_words = [
                c for c in filtered_words
                if pseg.lcut(c[0]) and re.match(postag, pseg.lcut(c[0])[0].flag)
            ]

        if min_change is not None:
            filtered_words = [c for c in filtered_words if c[1] >= min_change]

        filtered_changes[transition] = filtered_words

    return filtered_changes


def write_transition_csv(
    filepath: str,
    transition: str,
    word_changes: List[Tuple[str, float]],
    model: TempRefWord2Vec,
    cooc_count: Dict[str, Counter],
    stopwords: set,
    top_n: int
):
    """Write one CSV file for a single transition's semantic changes."""
    periods = transition.split("_to_")
    if len(periods) != 2:
        return

    from_period, to_period = periods
    filtered = [(w, c) for w, c in word_changes if w not in stopwords][:top_n]

    from_counts = model.get_period_vocab_counts(from_period)
    to_counts = model.get_period_vocab_counts(to_period)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "word", "change_score",
            "count_pre", "count_post",
            "cooc_pre", "cooc_post"
        ])
        for i, (word, change) in enumerate(filtered, 1):
            writer.writerow([
                i, word, f"{change:.6f}",
                from_counts[word], to_counts[word],
                cooc_count[from_period][word], cooc_count[to_period][word]
            ])

    print(f"  Wrote {filepath} ({len(filtered)} words)")


def run_analysis(
    model: TempRefWord2Vec,
    cooc_count: Dict[str, Counter],
    stopwords: set,
    output_dir: str,
    image_dir: str,
    target_word: str = "interiority",
    top_n: int = 100,
    postag: Optional[str] = None,
    generate_plots: bool = True,
    top_vocab: int = 1000,
    min_word_length: int = 2,
    min_cooc: int = 5,
    min_count: Tuple[int, int] = (5, 5),
    min_change: float = 0.0
):
    """Run single-model semantic change analysis and save results as CSV files."""
    changes = model.calculate_semantic_change(target_word)

    if generate_plots:
        from visualization_functions import plot_semantic_change_distribution, plot_temporal_similarity_heatmap
        os.makedirs(image_dir, exist_ok=True)
        plot_semantic_change_distribution(changes, target_word, bin_size=0.01, save_dir=image_dir)
        plot_temporal_similarity_heatmap(model, target_word, save_dir=image_dir)

    filtered_changes = filter_semantic_changes(
        changes, model,
        top_vocab=top_vocab,
        min_word_length=min_word_length,
        min_cooc=min_cooc,
        cooc_count=cooc_count,
        min_count=min_count,
        postag=postag,
        min_change=min_change
    )

    os.makedirs(output_dir, exist_ok=True)

    print("\nWriting per-transition CSVs...")
    for transition, word_changes in filtered_changes.items():
        csv_path = os.path.join(output_dir, f"semantic_changes_{transition}.csv")
        write_transition_csv(
            csv_path, transition, word_changes,
            model, cooc_count, stopwords, top_n
        )

    print(f"\nAnalysis complete. Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Multi-model ensemble mode
# ---------------------------------------------------------------------------

def discover_models(model_dir: str, include_null: bool = False) -> List[str]:
    """
    Find model .npy files in a directory.

    By default, excludes *_null.npy files (null distribution models).
    Set include_null=True to include only *_null.npy files.
    """
    all_npy = sorted(glob.glob(os.path.join(model_dir, "model_e*_s*.npy")))

    if include_null:
        return [p for p in all_npy if p.endswith("_null.npy")]
    else:
        return [p for p in all_npy if not p.endswith("_null.npy")]


def compute_pvalue(observed_mean: float, null_scores: np.ndarray) -> float:
    """
    One-sided p-value with finite-sample correction.

    Uses the standard Monte Carlo p-value formula: (1 + count) / (N + 1)
    This avoids impossible p=0 and gives less biased finite-sample estimates.
    With 100 null trials, minimum p-value is ~0.0099.
    """
    if len(null_scores) == 0:
        return np.nan
    return (1 + np.sum(null_scores >= observed_mean)) / (len(null_scores) + 1)


def filter_word(
    word: str,
    from_period: str,
    to_period: str,
    word_counts: Optional[Dict[str, Counter]],
    cooc_count: Optional[Dict[str, Counter]],
    min_word_length: int,
    min_cooc: int,
    min_count: int,
    postag: Optional[str]
) -> bool:
    """Check if a word passes all filters. Returns True if word should be kept."""
    if len(word) < min_word_length:
        return False

    if postag is not None:
        pos_result = pseg.lcut(word)
        if not pos_result or not re.match(postag, pos_result[0].flag):
            return False

    if min_count > 0 and word_counts is not None:
        if word_counts[from_period][word] < min_count or word_counts[to_period][word] < min_count:
            return False

    if min_cooc > 0 and cooc_count is not None:
        if cooc_count[to_period][word] < min_cooc:
            return False

    return True


def collect_scores_from_models(
    model_paths: List[str],
    target_word: str
) -> Dict[str, Dict[str, List[float]]]:
    """
    Load each model, compute semantic change, and collect per-word scores.

    Returns:
        {transition: {word: [score_model1, score_model2, ...]}}
    """
    trial_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for model_path in tqdm(model_paths, desc="Loading models"):
        model = TempRefWord2Vec.load(model_path)
        changes = model.calculate_semantic_change(target_word)

        for transition, word_changes in changes.items():
            for word, score in word_changes:
                trial_scores[transition][word].append(score)

    return trial_scores


def run_ensemble_analysis(
    model_dir: str,
    output_dir: str,
    target_word: str,
    data_dir: str,
    words_file: str,
    null_dir: Optional[str] = None,
    min_word_length: int = 2,
    min_cooc: int = 5,
    min_count: int = 5,
    postag: Optional[str] = "n.*",
):
    """
    Run multi-model ensemble analysis.

    Loads all models from model_dir, computes semantic change scores for each,
    aggregates across models, and writes per-transition CSVs.
    Optionally loads null models from null_dir for p-value calculation.
    """
    # Discover models
    model_paths = discover_models(model_dir, include_null=False)

    # If no non-null models found, try null models (user pointed --model-dir at null dir)
    is_null_mode = False
    if not model_paths:
        model_paths = discover_models(model_dir, include_null=True)
        is_null_mode = True

    if not model_paths:
        print(f"Error: No model files found in {model_dir}")
        print(f"  Expected files matching pattern: model_e*_s*.npy")
        sys.exit(1)

    print(f"Found {len(model_paths)} model(s) in {model_dir}")
    if is_null_mode:
        print("  (null distribution models detected)")

    # Collect scores from all models
    print(f"\nComputing semantic change scores for target '{target_word}'...")
    trial_scores = collect_scores_from_models(model_paths, target_word)

    # Collect null scores if provided
    null_scores_by_transition: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if not is_null_mode and null_dir:
        null_paths = discover_models(null_dir, include_null=True)
        if not null_paths:
            # Also try non-null pattern in case user saved without _null suffix
            null_paths = discover_models(null_dir, include_null=False)
        if null_paths:
            print(f"\nLoading {len(null_paths)} null model(s) from {null_dir}...")
            null_trial_scores = collect_scores_from_models(null_paths, target_word)
            null_scores_by_transition = {}
            for transition, word_scores in null_trial_scores.items():
                null_scores_by_transition[transition] = {
                    word: np.array(scores) for word, scores in word_scores.items()
                }
            print(f"  Loaded null distribution for {len(null_scores_by_transition)} transitions")
        else:
            print(f"\nWarning: No null models found in {null_dir}")

    # Compute word counts and co-occurrences for filtering
    word_counts = compute_word_counts(data_dir)
    target_words = load_target_words(words_file)
    cooc_count = None
    if min_cooc > 0:
        period_sentences = load_period_sentences(data_dir)
        cooc_count = compute_cooccurrences(period_sentences, target_words)

    # Aggregate and write results
    os.makedirs(output_dir, exist_ok=True)
    print("\nAggregating scores and writing CSVs...")

    for transition, word_scores in trial_scores.items():
        csv_path = os.path.join(output_dir, f"semantic_changes_{transition}.csv")

        periods = transition.split("_to_")
        if len(periods) != 2:
            continue
        from_period, to_period = periods

        rows = []
        filtered_count = 0
        null_missing_count = 0

        for word, scores in word_scores.items():
            if not filter_word(word, from_period, to_period, word_counts, cooc_count,
                               min_word_length, min_cooc, min_count, postag):
                filtered_count += 1
                continue

            scores_arr = np.array(scores)
            n_trials = len(scores_arr)
            if n_trials < 2:
                continue

            mean_change = np.mean(scores_arr)
            std_change = np.std(scores_arr, ddof=1)
            ci_low = np.percentile(scores_arr, 2.5)
            ci_high = np.percentile(scores_arr, 97.5)
            z_score = mean_change / std_change if std_change > 0 else np.nan

            row = {
                'word': word,
                'mean_change': mean_change,
                'std_change': std_change,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_trials': n_trials,
                'z_score': z_score,
            }

            if not is_null_mode:
                row['count_pre'] = word_counts[from_period][word] if word_counts else 0
                row['count_post'] = word_counts[to_period][word] if word_counts else 0
                row['cooc_pre'] = cooc_count[from_period][word] if cooc_count else 0
                row['cooc_post'] = cooc_count[to_period][word] if cooc_count else 0

            if not is_null_mode and null_scores_by_transition and transition in null_scores_by_transition:
                if word in null_scores_by_transition[transition]:
                    null_scores = null_scores_by_transition[transition][word]
                    row['null_mean'] = np.mean(null_scores)
                    row['p_value'] = compute_pvalue(mean_change, null_scores)
                else:
                    null_missing_count += 1
                    row['null_mean'] = np.nan
                    row['p_value'] = np.nan

            rows.append(row)

        rows.sort(key=lambda x: x['mean_change'], reverse=True)

        # Determine columns based on mode
        if is_null_mode:
            fieldnames = ['word', 'mean_change', 'std_change', 'ci_low', 'ci_high', 'n_trials', 'z_score']
            float_fields = ['mean_change', 'std_change', 'ci_low', 'ci_high', 'z_score']
        else:
            fieldnames = ['word', 'mean_change', 'std_change', 'ci_low', 'ci_high',
                          'n_trials', 'z_score', 'count_pre', 'count_post', 'cooc_pre', 'cooc_post',
                          'null_mean', 'p_value']
            float_fields = ['mean_change', 'std_change', 'ci_low', 'ci_high', 'z_score', 'null_mean', 'p_value']

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row_formatted = {k: row[k] for k in fieldnames if k in row}
                for key in float_fields:
                    if key in row_formatted and isinstance(row_formatted[key], float):
                        if np.isnan(row_formatted[key]):
                            row_formatted[key] = "nan"
                        else:
                            row_formatted[key] = f"{row_formatted[key]:.6f}"
                writer.writerow(row_formatted)

        stats = f"{len(rows)} words"
        if filtered_count > 0:
            stats += f", {filtered_count} filtered"
        if null_missing_count > 0:
            stats += f", {null_missing_count} missing from null"
        print(f"  Wrote {csv_path} ({stats})")

    print(f"\nEnsemble analysis complete. Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run semantic change analysis from trained TempRefWord2Vec model(s)."
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", default=None,
        help="Path to a single trained TempRefWord2Vec model (.npy)"
    )
    model_group.add_argument(
        "--model-dir", default=None,
        help="Directory of trained models for ensemble analysis (e.g. models/real/)"
    )

    # Null distribution (only for ensemble mode)
    parser.add_argument(
        "--null-dir", default=None,
        help="Directory of null (permuted) models for p-value calculation (ensemble mode only)"
    )

    # Data and words
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target replacement token in the model (default: interiority)"
    )

    # Output
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for CSV results (default: results)"
    )
    parser.add_argument(
        "--image-dir", default="images",
        help="Output directory for plots (single-model mode only, default: images)"
    )

    # Filtering options
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of top words per transition in single-model mode (default: 100)"
    )
    parser.add_argument(
        "--postag", default="n.*",
        help="POS tag regex filter (default: n.* for nouns, use 'none' to disable)"
    )
    parser.add_argument(
        "--top-vocab", type=int, default=1000,
        help="Keep words in top N most frequent vocab (single-model mode, default: 1000)"
    )
    parser.add_argument(
        "--min-word-length", type=int, default=2,
        help="Minimum word length to include (default: 2)"
    )
    parser.add_argument(
        "--min-cooc", type=int, default=5,
        help="Minimum co-occurrence count with target words (default: 5)"
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum word count in both periods (default: 5)"
    )
    parser.add_argument(
        "--min-change", type=float, default=0.0,
        help="Minimum change score threshold (single-model mode, default: 0.0)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots (single-model mode only)"
    )

    args = parser.parse_args()

    postag = args.postag if args.postag.lower() != 'none' else None

    if args.model:
        # --- Single-model mode ---
        print(f"Loading model from {args.model}...")
        model = TempRefWord2Vec.load(args.model)
        print(f"  Labels: {model.labels}")
        print(f"  Targets: {model.targets}")

        if args.null_dir:
            print("Warning: --null-dir is ignored in single-model mode")

        print(f"\nLoading segmented data from {args.data_dir}...")
        period_sentences_dict = load_period_sentences(args.data_dir)

        target_words = load_target_words(args.words)
        cooc_count = compute_cooccurrences(period_sentences_dict, target_words)
        stopwords = load_stopwords("zh_sim")

        run_analysis(
            model=model,
            cooc_count=cooc_count,
            stopwords=stopwords,
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            target_word=args.target,
            top_n=args.top_n,
            postag=postag,
            generate_plots=not args.no_plots,
            top_vocab=args.top_vocab,
            min_word_length=args.min_word_length,
            min_cooc=args.min_cooc,
            min_count=(args.min_count, args.min_count),
            min_change=args.min_change
        )

    else:
        # --- Multi-model ensemble mode ---
        run_ensemble_analysis(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            target_word=args.target,
            data_dir=args.data_dir,
            words_file=args.words,
            null_dir=args.null_dir,
            min_word_length=args.min_word_length,
            min_cooc=args.min_cooc,
            min_count=args.min_count,
            postag=postag,
        )


if __name__ == "__main__":
    main()
