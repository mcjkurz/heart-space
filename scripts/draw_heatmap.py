#!/usr/bin/env python3
"""Draw a temporal similarity heatmap for a specific word.

Usage (run from project root):
  python scripts/draw_heatmap.py
  python scripts/draw_heatmap.py --model models/tempref_stable_words.npy --word 声响
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from visualization_functions import plot_temporal_similarity_heatmap


def main():
    parser = argparse.ArgumentParser(description="Draw a temporal similarity heatmap.")
    parser.add_argument(
        "--model", default="models/tempref_interiority_w2v.npy",
        help="Path to a TempRefWord2Vec model file"
    )
    parser.add_argument(
        "--word", default="interiority",
        help="Target word to visualize (default: interiority)"
    )
    parser.add_argument(
        "--output-dir", default="images",
        help="Directory to save the heatmap (default: images)"
    )
    args = parser.parse_args()

    plot_temporal_similarity_heatmap(args.model, args.word, save_dir=args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
