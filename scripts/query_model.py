#!/usr/bin/env python3
"""Query semantic change for a specific word across periods.

Usage (run from project root):
  python scripts/query_model.py --query 大人
  python scripts/query_model.py --query 大人 --from mingqing --to late_qing
  python scripts/query_model.py --model models/tempref_stable_words.npy --target 声响 --query 耳朵
"""
import argparse
import os

from qhchina.analytics.word2vec import TempRefWord2Vec
from qhchina.utils import LineSentenceFile


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']


def query_single_transition(model, data_dir, target, query, period1, period2):
    """Query similarity change between two specific periods."""
    sentences1 = LineSentenceFile(os.path.join(data_dir, f"sentences_{period1}.txt"))
    sentences2 = LineSentenceFile(os.path.join(data_dir, f"sentences_{period2}.txt"))

    count1 = sum(s.count(query) for s in sentences1)
    count2 = sum(s.count(query) for s in sentences2)

    print(f"'{query}' appears {count1} times in {period1}")
    print(f"'{query}' appears {count2} times in {period2}")

    variant1 = f"{target}_{period1}"
    variant2 = f"{target}_{period2}"

    sim1 = model.similarity(variant1, query)
    sim2 = model.similarity(variant2, query)

    print(f"Similarity to {variant1}: {sim1:.4f}")
    print(f"Similarity to {variant2}: {sim2:.4f}")
    print(f"Change ({period2} - {period1}): {sim2 - sim1:+.4f}")


def query_all_periods(model, data_dir, target, query):
    """Query similarity change across all periods."""
    print(f"Querying '{query}' similarity to '{target}' across all periods\n")
    print(f"{'Period':<15} {'Count':>8} {'Similarity':>12}")
    print("-" * 40)

    similarities = []
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if not os.path.exists(filepath):
            print(f"{period:<15} {'(no data)':>8}")
            continue

        sentences = LineSentenceFile(filepath)
        count = sum(s.count(query) for s in sentences)

        variant = f"{target}_{period}"
        try:
            sim = model.similarity(variant, query)
            similarities.append((period, sim))
            print(f"{period:<15} {count:>8} {sim:>12.4f}")
        except KeyError:
            print(f"{period:<15} {count:>8} {'(not in vocab)':>12}")

    if len(similarities) > 1:
        print("\n" + "-" * 40)
        print("Consecutive changes:")
        for i in range(len(similarities) - 1):
            p1, s1 = similarities[i]
            p2, s2 = similarities[i + 1]
            print(f"  {p1} -> {p2}: {s2 - s1:+.4f}")

        first_period, first_sim = similarities[0]
        last_period, last_sim = similarities[-1]
        print(f"\nTotal change ({first_period} -> {last_period}): {last_sim - first_sim:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Query semantic change for a word across periods."
    )
    parser.add_argument(
        "--model", default="models/tempref_interiority_w2v.npy",
        help="Path to TempRefWord2Vec model file (default: models/tempref_interiority_w2v.npy)"
    )
    parser.add_argument(
        "--data-dir", default="data/segmented",
        help="Directory containing per-period sentence files (default: data/segmented)"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target word in the model (default: interiority)"
    )
    parser.add_argument(
        "--query", required=True,
        help="Word to query similarity change for"
    )
    parser.add_argument("--from", dest="period1", default=None, help="Source period (optional)")
    parser.add_argument("--to", dest="period2", default=None, help="Target period (optional)")
    args = parser.parse_args()

    model = TempRefWord2Vec.load(args.model)

    if args.period1 and args.period2:
        query_single_transition(model, args.data_dir, args.target, args.query, args.period1, args.period2)
    else:
        query_all_periods(model, args.data_dir, args.target, args.query)


if __name__ == "__main__":
    main()
