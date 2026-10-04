#!/usr/bin/env python3
"""Draw a temporal similarity heatmap for a specific word.

Supports single model or multi-model mode (averaging across models with std).

Usage (run from project root):
  # Single model
  python scripts/draw_heatmap.py --model models/real/model_e3_s42.npy

  # Multi-model (averages across all models, shows mean ± std)
  python scripts/draw_heatmap.py --model-dir models/real/
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    import matplotlib.pyplot as plt
    HAS_SEABORN = False

from qhchina.analytics import TempRefWord2Vec


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']
PERIOD_LABELS = ['Ming-Qing', 'Late Qing', 'Republican', 'Socialist', 'Contemporary']


def compute_similarity_matrix(model: TempRefWord2Vec, target_word: str) -> np.ndarray:
    """Compute pairwise similarity matrix for a target word's temporal variants."""
    n = len(PERIODS)
    matrix = np.zeros((n, n))

    for i, period_i in enumerate(PERIODS):
        for j, period_j in enumerate(PERIODS):
            if i == j:
                matrix[i, j] = 1.0
            else:
                variant_i = f"{target_word}_{period_i}"
                variant_j = f"{target_word}_{period_j}"
                try:
                    matrix[i, j] = model.similarity(variant_i, variant_j)
                except KeyError:
                    matrix[i, j] = np.nan

    return matrix


def plot_heatmap_single(model_path: str, target_word: str, output_dir: str, font_size: int):
    """Plot heatmap from a single model."""
    print(f"Loading model from {model_path}...")
    model = TempRefWord2Vec.load(model_path)
    matrix = compute_similarity_matrix(model, target_word)

    _plot_heatmap(matrix, None, target_word, output_dir, n_models=1, font_size=font_size)


def plot_heatmap_multi(model_dir: str, target_word: str, output_dir: str, font_size: int):
    """Plot heatmap averaging across multiple models, showing mean ± std."""
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.npy")))
    model_files = [f for f in model_files if "_null" not in os.path.basename(f)]

    if not model_files:
        print(f"No model files found in {model_dir}")
        return

    print(f"Found {len(model_files)} models in {model_dir}")

    matrices = []
    for i, path in enumerate(model_files):
        print(f"  [{i+1}/{len(model_files)}] Loading {os.path.basename(path)}...")
        model = TempRefWord2Vec.load(path)
        matrix = compute_similarity_matrix(model, target_word)
        matrices.append(matrix)

    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(stacked, axis=0)
    std_matrix = np.nanstd(stacked, axis=0)

    _plot_heatmap(mean_matrix, std_matrix, target_word, output_dir, n_models=len(model_files), font_size=font_size)


def _plot_heatmap(
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray | None,
    target_word: str,
    output_dir: str,
    n_models: int,
    font_size: int
):
    """Plot and save the heatmap."""
    n = len(PERIODS)
    periods_display = PERIOD_LABELS[::-1]
    mean_display = np.flipud(mean_matrix)
    std_display = np.flipud(std_matrix) if std_matrix is not None else None

    tick_size = font_size
    cell_size = font_size
    cbar_size = int(font_size * 0.9)

    plt.figure(figsize=(10, 8))

    if HAS_SEABORN:
        ax = sns.heatmap(
            mean_display,
            xticklabels=PERIOD_LABELS,
            yticklabels=periods_display,
            annot=False,
            cmap='YlOrRd',
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'},
            mask=np.isnan(mean_display)
        )

        for i in range(n):
            for j in range(n):
                val = mean_display[i, j]
                if np.isnan(val):
                    continue

                if std_display is not None:
                    std_val = std_display[i, j]
                    text = f"{val:.3f}\n±{std_val:.3f}"
                else:
                    text = f"{val:.3f}"

                color = 'white' if val > 0.7 else 'black'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        fontsize=cell_size, color=color)

        plt.xticks(rotation=0, fontsize=tick_size)
        plt.yticks(rotation=0, fontsize=tick_size)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=cbar_size)
        cbar.set_label('Cosine Similarity', fontsize=tick_size)
    else:
        im = plt.imshow(mean_display, cmap='YlOrRd', vmin=0.0, vmax=1.0, aspect='equal')
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=cbar_size)
        cbar.set_label('Cosine Similarity', fontsize=tick_size)

        for i in range(n):
            for j in range(n):
                val = mean_display[i, j]
                if np.isnan(val):
                    continue

                if std_display is not None:
                    std_val = std_display[i, j]
                    text = f"{val:.3f}\n±{std_val:.3f}"
                else:
                    text = f"{val:.3f}"

                color = 'white' if val > 0.7 else 'black'
                plt.text(j, i, text, ha='center', va='center', fontsize=cell_size, color=color, fontweight='normal')

        plt.xticks(range(n), PERIOD_LABELS, rotation=0, fontsize=tick_size)
        plt.yticks(range(n), periods_display, rotation=0, fontsize=tick_size)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_n{n_models}" if n_models > 1 else ""
    filename = f"temporal_similarity_heatmap_{target_word}{suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {filepath}")

    valid = mean_matrix[~np.isnan(mean_matrix)]
    if len(valid) > 0:
        print(f"\nSimilarity stats:")
        print(f"  Mean: {np.mean(valid):.4f}")
        print(f"  Min:  {np.min(valid):.4f}")
        print(f"  Max:  {np.max(valid):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw a temporal similarity heatmap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/draw_heatmap.py --model models/real/model_e3_s42.npy
  python scripts/draw_heatmap.py --model-dir models/real/
  python scripts/draw_heatmap.py --model-dir models/real/ --word 声响
"""
    )
    parser.add_argument(
        "--model",
        help="Path to a single TempRefWord2Vec model file"
    )
    parser.add_argument(
        "--model-dir",
        help="Directory containing multiple models (averages across all)"
    )
    parser.add_argument(
        "--word", default="interiority",
        help="Target word to visualize (default: interiority)"
    )
    parser.add_argument(
        "--output-dir", default="images",
        help="Directory to save the heatmap (default: images)"
    )
    parser.add_argument(
        "--font-size", type=int, default=12,
        help="Base font size; other sizes scale from this (default: 12)"
    )
    args = parser.parse_args()

    if args.model and args.model_dir:
        parser.error("Specify either --model or --model-dir, not both")

    if not args.model and not args.model_dir:
        parser.error("Must specify either --model or --model-dir")

    if args.model:
        plot_heatmap_single(args.model, args.word, args.output_dir, args.font_size)
    else:
        plot_heatmap_multi(args.model_dir, args.word, args.output_dir, args.font_size)

    print("Done!")


if __name__ == "__main__":
    main()
