#!/usr/bin/env python3
"""
Train separate Word2Vec models per period and compare similar words.

Also loads an existing TempRefWord2Vec model (if available) to compare
temporal variant similarities.

Usage (run from project root):
  python scripts/train_period_models.py
  python scripts/train_period_models.py --replacement interiority --top-n 50
  python scripts/train_period_models.py --tempref-model models/tempref_interiority_w2v.npy
"""

import argparse
import csv
import os
from collections import Counter
from typing import Dict, Iterable, List

from tqdm.auto import tqdm
from qhchina.analytics.word2vec import Word2Vec, TempRefWord2Vec
from qhchina.utils import LineSentenceFile


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']


class ReplacingLineSentenceFile:
    """Wrapper around LineSentenceFile that replaces target words on-the-fly."""

    def __init__(self, filepath: str, replacements: Dict[str, str]):
        self.filepath = filepath
        self.replacements = replacements

    def __iter__(self) -> Iterable[List[str]]:
        for sentence in LineSentenceFile(self.filepath):
            yield [self.replacements.get(word, word) for word in sentence]

MODEL_PARAMS = {
    "vector_size": 200,
    "window": 10,
    "min_word_count": 3,
    "sg": 1,
    "negative": 10,
    "alpha": 0.025,
    "seed": 42,
    "epochs": 3
}


def load_target_words(words_file):
    """Load target words from a text file (one word per line)."""
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


def load_period_sentences(data_dir, replacements=None):
    """Load segmented sentences from per-period .txt files with optional replacement."""
    corpora = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            if replacements:
                corpora[period] = ReplacingLineSentenceFile(filepath, replacements)
            else:
                corpora[period] = LineSentenceFile(filepath)
    print(f"Loaded {len(corpora)} periods: {list(corpora.keys())}")
    return corpora


def compute_word_frequencies(period_sentences):
    """Compute word frequencies per period."""
    print("\nComputing word frequencies...")
    freqs = {}
    for period in PERIODS:
        if period in period_sentences:
            freqs[period] = Counter()
            for sentence in tqdm(period_sentences[period], desc=f"  {period}", leave=False):
                freqs[period].update(sentence)
    return freqs


def train_or_load_period_model(period, sentences, models_dir, params):
    """Train a Word2Vec model for one period, or load if it already exists."""
    model_path = os.path.join(models_dir, f"{period}_w2v.npy")

    if os.path.exists(model_path):
        print(f"  Loading existing model for {period}...")
        return Word2Vec.load(model_path)

    print(f"  Training Word2Vec for {period}...")
    model = Word2Vec(sentences=sentences, **params)
    model.train()
    model.save(model_path)
    print(f"  Saved to {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train per-period Word2Vec models and compare similar words."
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
    parser.add_argument(
        "--models-dir", default="models/period_models",
        help="Directory for per-period model files"
    )
    parser.add_argument(
        "--tempref-model", default="models/tempref_interiority_w2v.npy",
        help="Path to TempRefWord2Vec model for temporal variant comparison"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for CSV results"
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of similar words to show per period (default: 50)"
    )
    parser.add_argument(
        "--min-freq", type=int, default=10,
        help="Minimum word frequency in the respective period to include (default: 10)"
    )
    args = parser.parse_args()

    params = MODEL_PARAMS.copy()

    target_words = load_target_words(args.words)
    print(f"Loaded {len(target_words)} target words, replacement: '{args.replacement}'")
    replacements = {word: args.replacement for word in target_words}

    period_sentences_raw = load_period_sentences(args.data_dir)
    period_sentences = load_period_sentences(args.data_dir, replacements=replacements)
    period_freqs = compute_word_frequencies(period_sentences_raw)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    period_results = {}

    for period in PERIODS:
        if period not in period_sentences:
            print(f"  {period} not found, skipping...")
            continue

        model = train_or_load_period_model(
            period, period_sentences[period], args.models_dir, params
        )

        print(f"\n{'='*50}")
        print(f"TOP {args.top_n} SIMILAR TO '{args.replacement}' IN {period.upper()}")
        print(f"{'='*50}")

        if args.replacement in model:
            # Request more results to account for filtering
            similar = model.most_similar(args.replacement, topn=args.top_n * 5)
            # Filter by minimum frequency in this period
            similar = [
                (word, sim) for word, sim in similar
                if period_freqs[period].get(word, 0) >= args.min_freq
            ][:args.top_n]
            period_results[period] = []
            for i, (word, sim) in enumerate(similar, 1):
                freq_str = ", ".join(
                    f"{p[:3]}:{period_freqs[p].get(word, 0)}" for p in PERIODS
                )
                print(f"  {i:2}. {word}: {sim:.4f} ({freq_str})")
                row = {"rank": i, "word": word, "similarity": sim}
                for p in PERIODS:
                    row[f"freq_{p}"] = period_freqs[p].get(word, 0)
                period_results[period].append(row)
        else:
            print(f"  '{args.replacement}' not found in vocabulary")
        print()

    csv_path = os.path.join(args.output_dir, "period_model_similar_words.csv")
    fieldnames = ["period", "rank", "word", "similarity"] + [f"freq_{p}" for p in PERIODS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for period in PERIODS:
            for row in period_results.get(period, []):
                writer.writerow({"period": period, **row})
    print(f"Period model results saved to {csv_path}")

    # Temporal variant comparison using TempRef model
    if os.path.exists(args.tempref_model):
        print("\n" + "=" * 50)
        print("TEMPREF MODEL COMPARISON")
        print("=" * 50)
        tempref = TempRefWord2Vec.load(args.tempref_model)

        tempref_results = []
        for period in PERIODS:
            variant = f"{args.replacement}_{period}"
            try:
                similar = tempref.most_similar(variant, topn=args.top_n * 5)
            except KeyError:
                print(f"  {variant}: not found in model")
                continue
            # Filter by minimum frequency in this period
            similar = [
                (word, sim) for word, sim in similar
                if period_freqs[period].get(word, 0) >= args.min_freq
            ][:args.top_n]
            print(f"\nTop {args.top_n} similar to {variant}:")
            for i, (word, sim) in enumerate(similar, 1):
                freq_str = ", ".join(
                    f"{p[:3]}:{period_freqs[p].get(word, 0)}" for p in PERIODS
                )
                print(f"  {word}: {sim:.4f} ({freq_str})")
                row = {"period": period, "rank": i, "word": word, "similarity": sim}
                for p in PERIODS:
                    row[f"freq_{p}"] = period_freqs[p].get(word, 0)
                tempref_results.append(row)

        tempref_csv = os.path.join(args.output_dir, "tempref_similar_words.csv")
        with open(tempref_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in tempref_results:
                writer.writerow(row)
        print(f"\nTempRef results saved to {tempref_csv}")
    else:
        print(f"\nTempRef model not found at {args.tempref_model}, skipping comparison.")


if __name__ == "__main__":
    main()
