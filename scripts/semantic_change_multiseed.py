#!/usr/bin/env python3
"""
Multi-seed stability analysis for semantic change scores.

Estimates how stable semantic change scores are across different random seeds
(training variance), and builds a null distribution via period permutation
for significance testing.

NOTE: This is NOT bootstrap (which would resample the corpus). Multi-seed analysis
measures sensitivity to Word2Vec's stochastic training (initialization, negative
sampling, batch order), not sensitivity to corpus sampling.

Usage (run from project root):
  # Multi-seed mode (multiple seeds, real labels)
  python scripts/semantic_change_multiseed.py --trials 100 --output-dir results/multiseed

  # Null distribution mode (shuffled period labels)
  python scripts/semantic_change_multiseed.py --trials 100 --permute-periods --output-dir results/null_distribution
"""

import argparse
import csv
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import jieba.posseg as pseg
import numpy as np
from tqdm.auto import tqdm

from qhchina.analytics import TempRefWord2Vec
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
    "calculate_loss": False,
    "workers": 4,
    "sampling_strategy": "balanced",
}


class ReplacingLineSentenceFile:
    """Wrapper around LineSentenceFile that replaces target words on-the-fly."""

    def __init__(self, filepath: str, replacements: Dict[str, str]):
        self.filepath = filepath
        self.replacements = replacements

    def __iter__(self):
        for sentence in LineSentenceFile(self.filepath):
            yield [self.replacements.get(word, word) for word in sentence]


def load_target_words(words_file: str) -> List[str]:
    """Load target words from a text file (one word per line)."""
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Target words file not found: {words_file}")
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


def build_sentence_index(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """Build global sentence index: [(period, line_idx), ...] and period sizes."""
    sentence_index = []
    period_sizes = {}
    
    print("Building sentence index...")
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                n_lines = sum(1 for _ in f)
            sentence_index.extend([(period, i) for i in range(n_lines)])
            period_sizes[period] = n_lines
            print(f"  {period}: {n_lines:,} sentences")
    
    print(f"Total: {len(sentence_index):,} sentences")
    return sentence_index, period_sizes


def create_permuted_corpus_files(
    data_dir: str,
    temp_dir: str,
    sentence_index: List[Tuple[str, int]],
    period_sizes: Dict[str, int],
    seed: int
) -> Dict[str, str]:
    """Create temporary permuted sentence files, return paths."""
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
    
    temp_paths = {}
    for period, assigned_indices in assignments.items():
        temp_path = os.path.join(temp_dir, f"sentences_{period}.txt")
        with open(temp_path, 'w', encoding='utf-8') as f:
            for src_period, line_idx in assigned_indices:
                f.write(source_files[src_period][line_idx])
        temp_paths[period] = temp_path
    
    return temp_paths


def load_period_sentences(
    data_dir: str,
    replacements: Dict[str, str]
) -> Dict[str, ReplacingLineSentenceFile]:
    """Load segmented sentences from per-period .txt files with word replacement."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            corpora[period] = ReplacingLineSentenceFile(filepath, replacements)
    return corpora


def compute_word_counts(data_dir: str) -> Dict[str, Counter]:
    """Compute word counts per period."""
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


def compute_cooccurrences(
    data_dir: str,
    target_words: List[str],
    window_size: int = 10
) -> Dict[str, Counter]:
    """Compute co-occurrence counts for target words within a window."""
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


def train_and_compute_changes(
    data_dir: str,
    replacements: Dict[str, str],
    target: str,
    params: dict
) -> Dict[str, List[Tuple[str, float]]]:
    """Train model and compute semantic changes, return all word changes."""
    corpora = load_period_sentences(data_dir, replacements)
    
    model = TempRefWord2Vec(
        sentences=corpora,
        targets=[target],
        **params,
    )
    model.train()
    
    return model.calculate_semantic_change(target)


def save_checkpoint(
    trial_scores: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    filename: str = "trial_scores.npz"
):
    """Save trial scores to npz file."""
    save_dict = {}
    for transition, word_scores in trial_scores.items():
        for word, scores in word_scores.items():
            key = f"{transition}|{word}"
            save_dict[key] = np.array(scores)
    
    np.savez_compressed(os.path.join(output_dir, filename), **save_dict)


def load_checkpoint(filepath: str) -> Dict[str, Dict[str, List[float]]]:
    """Load trial scores from npz file."""
    trial_scores = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(filepath):
        return trial_scores
    
    data = np.load(filepath, allow_pickle=True)
    for key in data.files:
        transition, word = key.split("|", 1)
        trial_scores[transition][word] = data[key].tolist()
    
    return trial_scores


def load_null_distribution(null_dir: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Load null distribution scores for p-value calculation."""
    null_path = os.path.join(null_dir, "null_scores.npz")
    if not os.path.exists(null_path):
        return None
    
    null_scores = defaultdict(dict)
    data = np.load(null_path, allow_pickle=True)
    for key in data.files:
        transition, word = key.split("|", 1)
        null_scores[transition][word] = data[key]
    
    return null_scores


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
    word_counts: Dict[str, Counter],
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


def aggregate_and_save_results(
    trial_scores: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    is_null_mode: bool = False,
    null_dir: Optional[str] = None,
    word_counts: Optional[Dict[str, Counter]] = None,
    cooc_count: Optional[Dict[str, Counter]] = None,
    min_word_length: int = 2,
    min_cooc: int = 5,
    min_count: int = 5,
    postag: Optional[str] = "n.*"
):
    """Aggregate trial scores, apply filters, and save per-transition CSVs."""
    null_distribution = None
    if not is_null_mode and null_dir:
        null_distribution = load_null_distribution(null_dir)
        if null_distribution:
            print(f"Loaded null distribution from {null_dir}")
    
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
            
            if not is_null_mode and null_distribution and transition in null_distribution:
                if word in null_distribution[transition]:
                    null_scores = null_distribution[transition][word]
                    row['null_mean'] = np.mean(null_scores)
                    row['p_value'] = compute_pvalue(mean_change, null_scores)
                else:
                    null_missing_count += 1
                    row['null_mean'] = np.nan
                    row['p_value'] = np.nan
            
            rows.append(row)
        
        rows.sort(key=lambda x: x['mean_change'], reverse=True)
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple semantic change trials for statistical analysis."
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of training trials (default: 100)"
    )
    parser.add_argument(
        "--permute-periods", action="store_true",
        help="Shuffle sentences across period labels (null distribution mode)"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/bootstrap or results/null_distribution)"
    )
    parser.add_argument(
        "--null-dir", default="results/null_distribution",
        help="Path to null distribution for p-value calculation"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file"
    )
    parser.add_argument(
        "--replacement", default="interiority",
        help="Replacement token for target words"
    )
    parser.add_argument(
        "--seed-start", type=int, default=42,
        help="Starting seed (seeds = seed_start + trial_index)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10,
        help="Save checkpoint every N trials (default: 10)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint if available"
    )
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--model-min-count", type=int, default=None,
                       help="Minimum word count for model vocabulary (default: 3)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    
    parser.add_argument(
        "--min-word-length", type=int, default=2,
        help="Minimum word length to include in results (default: 2)"
    )
    parser.add_argument(
        "--min-cooc", type=int, default=5,
        help="Minimum co-occurrence count with target words in target period (default: 5)"
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum word count in EACH period of transition (default: 5)"
    )
    parser.add_argument(
        "--postag", default="n.*",
        help="POS tag regex filter (default: n.* for nouns, use 'none' to disable)"
    )
    parser.add_argument(
        "--skip-csv", action="store_true",
        help="Only save checkpoint, skip CSV generation (for parallel runs)"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = "results/null_distribution" if args.permute_periods else "results/multiseed"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    params = MODEL_PARAMS.copy()
    if args.vector_size is not None:
        params["vector_size"] = args.vector_size
    if args.window is not None:
        params["window"] = args.window
    if args.model_min_count is not None:
        params["min_word_count"] = args.model_min_count
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.workers is not None:
        params["workers"] = args.workers
    
    target_words = load_target_words(args.words)
    replacements = {word: args.replacement for word in target_words}
    print(f"Loaded {len(target_words)} target words, replacement: '{args.replacement}'")
    
    postag = args.postag if args.postag.lower() != 'none' else None
    
    word_counts = None
    cooc_count = None
    if not args.skip_csv:
        word_counts = compute_word_counts(args.data_dir)
        if args.min_cooc > 0:
            cooc_count = compute_cooccurrences(args.data_dir, target_words)
    
    mode = "null distribution (permuted)" if args.permute_periods else "multi-seed (real labels)"
    print(f"\nMode: {mode}")
    print(f"Trials: {args.trials}")
    print(f"Seeds: {args.seed_start} to {args.seed_start + args.trials - 1}")
    print(f"Output: {args.output_dir}")
    
    sentence_index = None
    period_sizes = None
    if args.permute_periods:
        sentence_index, period_sizes = build_sentence_index(args.data_dir)
    
    checkpoint_file = "null_scores.npz" if args.permute_periods else "multiseed_scores.npz"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_file)
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        trial_scores = load_checkpoint(checkpoint_path)
        if trial_scores:
            first_word_scores = next(iter(next(iter(trial_scores.values())).values()))
            start_trial = len(first_word_scores)
            print(f"  Found {start_trial} completed trials")
        else:
            start_trial = 0
    else:
        trial_scores = defaultdict(lambda: defaultdict(list))
        start_trial = 0
    
    print(f"\nStarting trials {start_trial} to {args.trials - 1}...")
    
    for trial in tqdm(range(start_trial, args.trials), desc="Trials"):
        seed = args.seed_start + trial
        params["seed"] = seed
        
        if args.permute_periods:
            temp_dir = tempfile.mkdtemp(prefix="permuted_corpus_")
            try:
                create_permuted_corpus_files(
                    args.data_dir, temp_dir,
                    sentence_index, period_sizes, seed
                )
                changes = train_and_compute_changes(
                    temp_dir, replacements, args.replacement, params
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            changes = train_and_compute_changes(
                args.data_dir, replacements, args.replacement, params
            )
        
        for transition, word_changes in changes.items():
            for word, score in word_changes:
                trial_scores[transition][word].append(score)
        
        if (trial + 1) % args.checkpoint_interval == 0:
            save_checkpoint(trial_scores, args.output_dir, checkpoint_file)
            tqdm.write(f"  Checkpoint saved at trial {trial + 1}")
    
    print("\nSaving final results...")
    save_checkpoint(trial_scores, args.output_dir, checkpoint_file)
    
    if args.skip_csv:
        print(f"\nDone! Checkpoint saved to {args.output_dir}/ (CSV generation skipped)")
    else:
        null_dir = args.null_dir if not args.permute_periods else None
        aggregate_and_save_results(
            trial_scores, args.output_dir,
            is_null_mode=args.permute_periods,
            null_dir=null_dir,
            word_counts=word_counts,
            cooc_count=cooc_count,
            min_word_length=args.min_word_length,
            min_cooc=args.min_cooc,
            min_count=args.min_count,
            postag=postag
        )
        print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
