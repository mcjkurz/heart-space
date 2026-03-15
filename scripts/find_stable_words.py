#!/usr/bin/env python3
"""
Find the most semantically stable words across time periods.

Identifies words that appear frequently in all periods, trains TempRef model(s)
with them as targets, and ranks them by semantic stability. Supports multiple
trials with different seeds, parallel execution, and checkpoint/resume.

Usage (run from project root):
  python scripts/find_stable_words.py
  python scripts/find_stable_words.py --trials 10 --processes 4
  python scripts/find_stable_words.py --min-freq 50 --trials 5 --resume
"""

import argparse
import os
import time
from multiprocessing import Pool

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
    "calculate_loss": False,
    "workers": 2,
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
    period_vocab = {}
    for period in PERIODS:
        if period not in period_sentences:
            continue
        counter = Counter()
        for sentence in period_sentences[period]:
            counter.update(sentence)
        period_vocab[period] = counter
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
    return candidates or set()


def calculate_stability_scores(model, target_words):
    """Calculate consecutive similarity scores for each target word."""
    scores = {}
    for word in target_words:
        consecutive_sims = []
        for i in range(len(PERIODS) - 1):
            variant1 = f"{word}_{PERIODS[i]}"
            variant2 = f"{word}_{PERIODS[i + 1]}"
            try:
                sim = model.similarity(variant1, variant2)
                consecutive_sims.append(sim)
            except Exception:
                consecutive_sims.append(np.nan)
        scores[word] = consecutive_sims
    return scores


def train_single_trial(args: tuple) -> dict:
    """
    Train a single TempRefWord2Vec model and return stability scores.

    This is a top-level function (required for multiprocessing pickling).
    Accepts a single tuple argument for compatibility with Pool.map().

    Returns:
        A dict mapping word -> list of consecutive similarities.
    """
    seed, corpora_paths, target_words, params = args

    trial_params = params.copy()
    trial_params["seed"] = seed

    corpora = {}
    for period, filepath in corpora_paths.items():
        corpora[period] = LineSentenceFile(filepath)

    model = TempRefWord2Vec(
        sentences=corpora,
        targets=target_words,
        **trial_params,
    )
    model.train()

    scores = calculate_stability_scores(model, target_words)
    return scores


def aggregate_trial_scores(all_trial_scores, target_words):
    """Aggregate scores across multiple trials."""
    aggregated = []
    n_transitions = len(PERIODS) - 1

    for word in target_words:
        consecutive_sims_per_trial = []
        for trial_scores in all_trial_scores:
            if word in trial_scores:
                consecutive_sims_per_trial.append(trial_scores[word])

        consecutive_sims_per_trial = np.array(consecutive_sims_per_trial)
        n_trials = len(consecutive_sims_per_trial)

        mean_sims = np.nanmean(consecutive_sims_per_trial, axis=1)
        mean_sim_mean = np.nanmean(mean_sims)
        mean_sim_std = np.nanstd(mean_sims)

        consec_means = []
        consec_stds = []
        for i in range(n_transitions):
            vals = consecutive_sims_per_trial[:, i]
            consec_means.append(np.nanmean(vals))
            consec_stds.append(np.nanstd(vals))

        aggregated.append({
            'word': word,
            'mean_sim_mean': mean_sim_mean,
            'mean_sim_std': mean_sim_std,
            'n_trials': n_trials,
            'consecutive_means': consec_means,
            'consecutive_stds': consec_stds,
        })

    aggregated.sort(key=lambda x: -x['mean_sim_mean'])
    return aggregated


def save_results(aggregated_scores, output_path):
    """Save aggregated stability scores to CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    transition_names = [f"{PERIODS[i]}_{PERIODS[i+1]}" for i in range(len(PERIODS)-1)]

    with open(output_path, 'w', encoding='utf-8') as f:
        headers = ["rank", "word", "n_trials", "mean_sim_mean", "mean_sim_std"]
        for name in transition_names:
            headers.append(f"sim_{name}_mean")
            headers.append(f"sim_{name}_std")
        f.write(",".join(headers) + "\n")

        for i, row in enumerate(aggregated_scores):
            values = [
                str(i + 1),
                row['word'],
                str(row['n_trials']),
                f"{row['mean_sim_mean']:.6f}",
                f"{row['mean_sim_std']:.6f}",
            ]
            for j in range(len(transition_names)):
                values.append(f"{row['consecutive_means'][j]:.6f}")
                values.append(f"{row['consecutive_stds'][j]:.6f}")
            f.write(",".join(values) + "\n")


def save_checkpoint(all_trial_scores, checkpoint_path):
    """Save trial scores to a checkpoint file."""
    save_dict = {
        'n_trials': len(all_trial_scores),
    }
    for trial_idx, trial_scores in enumerate(all_trial_scores):
        for word, sims in trial_scores.items():
            key = f"trial_{trial_idx}|{word}"
            save_dict[key] = np.array(sims)
    np.savez_compressed(checkpoint_path, **save_dict)


def load_checkpoint(checkpoint_path):
    """Load trial scores from a checkpoint file. Returns (all_trial_scores, n_trials)."""
    if not os.path.exists(checkpoint_path):
        return None, 0

    data = np.load(checkpoint_path, allow_pickle=True)
    n_trials = int(data['n_trials'])

    all_trial_scores = [{} for _ in range(n_trials)]
    for key in data.files:
        if key.startswith("trial_"):
            parts = key.split("|", 1)
            if len(parts) == 2:
                trial_idx = int(parts[0].replace("trial_", ""))
                word = parts[1]
                all_trial_scores[trial_idx][word] = data[key].tolist()

    return all_trial_scores, n_trials


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
        help="Output path for the final trained model (default: models/tempref_stable_words.npy)"
    )
    parser.add_argument(
        "--results", default="results/word_stability_scores.csv",
        help="Output path for stability scores CSV (default: results/word_stability_scores.csv)"
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
        "--trials", type=int, default=1,
        help="Number of training trials with different seeds (default: 1)"
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base seed; trial i uses seed base_seed + i (default: 42)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint if available"
    )
    parser.add_argument(
        "--processes", type=int, default=1,
        help="Number of parallel processes for training (default: 1)"
    )
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.min_freq < 1:
        parser.error("--min-freq must be at least 1")

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

    period_sentences = load_period_sentences(args.data_dir)
    print(f"Loaded {len(period_sentences)} periods", flush=True)
    period_vocab = compute_word_frequencies(period_sentences)
    print(f"Computed word frequencies for {len(period_vocab)} periods", flush=True)
    candidates = find_candidate_words(period_vocab, args.min_freq, args.min_word_length)
    print(f"Found {len(candidates)} candidate words meeting --min-freq {args.min_freq}", flush=True)
    target_words = sorted(candidates)
    corpora = {period: period_sentences[period] for period in PERIODS if period in period_sentences}
    print(f"Loaded {len(corpora)} periods for training", flush=True)

    checkpoint_path = args.results.replace(".csv", "_checkpoint.npz")
    all_trial_scores = []
    start_trial = 0

    if args.resume:
        loaded_scores, n_completed = load_checkpoint(checkpoint_path)
        if loaded_scores is not None:
            all_trial_scores = loaded_scores
            start_trial = n_completed
            print(f"Resumed from checkpoint: {n_completed} trials completed", flush=True)

    # Build corpora paths for multiprocessing (can't pickle LineSentenceFile iterators)
    corpora_paths = {}
    for period in PERIODS:
        filepath = os.path.join(args.data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            corpora_paths[period] = filepath

    # Build task arguments for remaining trials
    remaining_trials = args.trials - start_trial
    seeds = [args.base_seed + i for i in range(start_trial, args.trials)]

    def fmt_elapsed(start: float) -> str:
        """Format elapsed time as HH:MM:SS."""
        elapsed = int(time.time() - start)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    print(f"Training {remaining_trials} trials with seeds {seeds[0]} to {seeds[-1]}", flush=True)
    print(f"Processes: {args.processes}", flush=True)

    t0 = time.time()

    if args.processes <= 1 or remaining_trials == 1:
        # Sequential mode (supports checkpointing)
        for trial in range(start_trial, args.trials):
            seed = args.base_seed + trial
            print(f"Training trial {trial + 1}/{args.trials} with seed {seed}...", flush=True)

            task_args = (seed, corpora_paths, target_words, params)
            scores = train_single_trial(task_args)
            all_trial_scores.append(scores)

            save_checkpoint(all_trial_scores, checkpoint_path)
            print(f"  Checkpoint saved ({trial + 1}/{args.trials} trials) [{fmt_elapsed(t0)} elapsed]", flush=True)
    else:
        # Parallel mode
        task_args = [(seed, corpora_paths, target_words, params) for seed in seeds]
        print(f"Launching {args.processes} parallel processes...", flush=True)

        with Pool(processes=args.processes) as pool:
            for i, scores in enumerate(pool.imap_unordered(train_single_trial, task_args), 1):
                all_trial_scores.append(scores)
                save_checkpoint(all_trial_scores, checkpoint_path)
                print(f"  [{i}/{remaining_trials}] Trial complete [{fmt_elapsed(t0)} elapsed]", flush=True)

    # Train final model to save (using last seed)
    print(f"\nTraining final model (seed {args.base_seed + args.trials - 1}) to save...", flush=True)
    final_corpora = {period: LineSentenceFile(path) for period, path in corpora_paths.items()}
    final_params = params.copy()
    final_params["seed"] = args.base_seed + args.trials - 1
    model = TempRefWord2Vec(
        sentences=final_corpora,
        targets=target_words,
        **final_params,
    )
    model.train()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)

    total_elapsed = fmt_elapsed(t0)
    print(f"Done! {args.trials} trial(s) completed [total: {total_elapsed}]", flush=True)

    aggregated = aggregate_trial_scores(all_trial_scores, target_words)
    print(f"Aggregated {len(aggregated)} word scores", flush=True)
    save_results(aggregated, args.results)
    print(f"Results saved to {args.results}", flush=True)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
