#!/usr/bin/env python3
"""
Semantic change analysis using a trained TempRefWord2Vec model.

Loads a trained model, computes semantic changes for a target word,
filters results, and outputs one CSV per transition plus a summary CSV.

Usage (run from project root):
  python scripts/semantic_change.py
  python scripts/semantic_change.py --model models/tempref_interiority_w2v.npy --output-dir results/
  python scripts/semantic_change.py --target interiority --top-n 200 --postag "n.*" --no-plots
  python scripts/semantic_change.py --top-vocab 2000 --min-word-length 2 --min-cooc 10 --min-change 0.01
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import jieba.posseg as pseg
from tqdm.auto import tqdm

from qhchina.analytics import TempRefWord2Vec
from qhchina.helpers.texts import load_stopwords
from qhchina.utils import LineSentenceFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']

def load_target_words(words_file):
    """Load target words from a text file (one word per line)."""
    if not os.path.exists(words_file):
        raise FileNotFoundError(
            f"Target words file not found: {words_file}\n"
            f"Expected a text file with one word per line at data/dictionaries/interiority_words.txt"
        )
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


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


def write_similarity_csv(filepath: str, model: TempRefWord2Vec, target_word: str, top_n: int):
    """Write CSV of temporal variant similarities."""
    labels = model.get_time_labels()

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["period", "rank", "word", "similarity", "cross_space"])

        for label in labels:
            variant = f"{target_word}_{label}"
            for cross_space in [True, False]:
                try:
                    similar = model.most_similar(variant, topn=top_n, cross_space=cross_space)
                    for rank, (word, sim) in enumerate(similar, 1):
                        writer.writerow([label, rank, word, f"{sim:.6f}", cross_space])
                except KeyError:
                    pass

    print(f"  Wrote {filepath}")


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
    """Run semantic change analysis and save results as CSV files."""
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

    sim_csv_path = os.path.join(output_dir, "temporal_variant_similarities.csv")
    print("\nWriting similarity CSV...")
    write_similarity_csv(sim_csv_path, model, target_word, top_n)

    print(f"\nAnalysis complete. Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run semantic change analysis and output results as CSV."
    )
    parser.add_argument(
        "--model", default="models/tempref_interiority_w2v.npy",
        help="Path to trained TempRefWord2Vec model"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to interiority words file"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for CSV results (default: results)"
    )
    parser.add_argument(
        "--image-dir", default="images",
        help="Output directory for plots (default: images)"
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of top words to include per transition (default: 100)"
    )
    parser.add_argument(
        "--postag", default="n.*",
        help="POS tag regex filter (default: n.* for nouns)"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target replacement token in the model (default: interiority)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--top-vocab", type=int, default=1000,
        help="Keep words in top N most frequent vocab of either period (default: 1000)"
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
        help="Minimum change score threshold (default: 0.0, only positive changes)"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = TempRefWord2Vec.load(args.model)
    print(f"  Labels: {model.labels}")
    print(f"  Targets: {model.targets}")

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
        postag=args.postag,
        generate_plots=not args.no_plots,
        top_vocab=args.top_vocab,
        min_word_length=args.min_word_length,
        min_cooc=args.min_cooc,
        min_count=(args.min_count, args.min_count),
        min_change=args.min_change
    )


if __name__ == "__main__":
    main()
