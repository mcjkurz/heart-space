#!/usr/bin/env python3
"""
3D PCA visualization of temporal word variants.

Loads a TempRefWord2Vec model, extracts vectors for temporal variants of a
target word, reduces to 3D with PCA, and plots the semantic trajectory.

Supports single model or batch mode (one plot per model in a directory).

Usage (run from project root):
  # Single model
  python scripts/visualize_pca_3d.py --model models/real/model_e3_s42.npy

  # Batch mode (one plot per model)
  python scripts/visualize_pca_3d.py --model-dir models/real/
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from qhchina.analytics import TempRefWord2Vec


PERIOD_ANNOTATIONS = {
    "mingqing": "Ming-Qing",
    "late_qing": "Late Qing",
    "republican": "Republican",
    "socialist": "Socialist",
    "contemporary": "Contemporary",
}

PERIOD_LEGEND_LABELS = {
    "mingqing": "Ming-Qing (1368 - 1860)",
    "late_qing": "Late Qing (1860 - 1911)",
    "republican": "Republican (1911 - 1949)",
    "socialist": "Socialist (1949 - 1976)",
    "contemporary": "Contemporary (1976 - )",
}

ANNOTATION_OFFSETS = {
    "mingqing": (-0.3, 0.3, 0.0),
    "late_qing": (-0.3, 0.3, 0.0),
    "republican": (0.6, -0.3, 0.6),
    "socialist": (0.0, 0.3, 0.0),
    "contemporary": (-0.3, 0.3, 0.0),
}
DEFAULT_OFFSET = (0.3, 0.3, 0.3)


def plot_temporal_3d(model_path, target_word="interiority", output_dir="images", output_suffix="", show=True, font_size=12):
    """Load model, extract temporal variants, PCA to 3D, and save a plot.
    
    Args:
        model_path: Path to the model file.
        target_word: Target word to visualize.
        output_dir: Directory to save the plot.
        output_suffix: Optional suffix for the output filename.
        show: Whether to display the plot interactively.
        font_size: Base font size; other sizes scale from this.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = TempRefWord2Vec.load(model_path)

    time_labels = model.get_time_labels()
    available_targets = model.get_available_targets()

    print(f"Available targets: {available_targets}")
    print(f"Time periods: {time_labels}")

    if target_word not in available_targets:
        print(f"Error: '{target_word}' not in available targets!")
        return

    vectors = []
    valid_variants = []
    valid_labels = []

    for label in time_labels:
        variant = f"{target_word}_{label}"
        try:
            vector = model.get_vector(variant)
            if vector is not None:
                vectors.append(vector)
                valid_variants.append(variant)
                valid_labels.append(label)
                print(f"  Found vector for '{variant}' (dim={vector.shape[0]})")
            else:
                print(f"  No vector for '{variant}'")
        except Exception as e:
            print(f"  Error for '{variant}': {e}")

    if len(vectors) < 2:
        print("Error: Need at least 2 temporal variants for PCA.")
        return

    vectors_array = np.array(vectors)
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors_array)
    explained = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: {explained} (total: {explained.sum():.3f})")

    annotation_size = font_size
    label_size = font_size
    legend_size = int(font_size * 0.9)
    tick_size = int(font_size * 0.8)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_variants)))

    ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
               c=colors, s=150, alpha=0.8)

    for i, label in enumerate(valid_labels):
        annotation = PERIOD_ANNOTATIONS.get(label, label)
        px, py, pz = vectors_3d[i]
        text = f'{annotation}\n({px:.2f}, {py:.2f}, {pz:.2f})'

        offset = np.array(ANNOTATION_OFFSETS.get(label, DEFAULT_OFFSET))
        pos = vectors_3d[i] + offset

        ax.text(pos[0], pos[1], pos[2], text,
                fontsize=annotation_size, fontweight='bold', ha='center', alpha=0.8)

    if len(vectors_3d) > 1:
        ax.plot(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
                'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', fontsize=label_size, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', fontsize=label_size, fontweight='bold')
    ax.set_zlabel(f'PC3 ({explained[2]:.1%} variance)', fontsize=label_size, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_size)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[i], markersize=10,
                   label=PERIOD_LEGEND_LABELS.get(label, label))
        for i, label in enumerate(valid_labels)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=legend_size)

    plt.tight_layout(pad=3.0)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    output_file = os.path.join(output_dir, f'temporal_{target_word}_3d_pca{suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"\nPlot saved: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"\nTarget: {target_word}")
    print(f"Variants: {len(valid_variants)}, PCA total variance: {explained.sum():.3f}")
    if len(vectors_3d) > 1:
        print("\nConsecutive distances (3D PCA space):")
        for i in range(len(vectors_3d) - 1):
            dist = np.linalg.norm(vectors_3d[i+1] - vectors_3d[i])
            print(f"  {valid_labels[i]} -> {valid_labels[i+1]}: {dist:.3f}")


def plot_batch(model_dir: str, target_word: str, output_dir: str, font_size: int):
    """Generate one PCA plot per model in the directory."""
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.npy")))
    model_files = [f for f in model_files if "_null" not in os.path.basename(f)]

    if not model_files:
        print(f"No model files found in {model_dir}")
        return

    print(f"Found {len(model_files)} models in {model_dir}\n")

    for i, path in enumerate(model_files):
        basename = os.path.basename(path).replace(".npy", "")
        print(f"[{i+1}/{len(model_files)}] Processing {basename}...")
        plot_temporal_3d(
            path,
            target_word=target_word,
            output_dir=output_dir,
            output_suffix=basename,
            show=False,
            font_size=font_size
        )
        print()

    print(f"Done! Generated {len(model_files)} plots in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="3D PCA visualization of temporal word variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_pca_3d.py --model models/real/model_e3_s42.npy
  python scripts/visualize_pca_3d.py --model-dir models/real/
  python scripts/visualize_pca_3d.py --model-dir models/real/ --target 声响
"""
    )
    parser.add_argument(
        "--model",
        help="Path to a single TempRefWord2Vec model file"
    )
    parser.add_argument(
        "--model-dir",
        help="Directory containing multiple models (generates one plot per model)"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target word to visualize (default: interiority)"
    )
    parser.add_argument(
        "--output-dir", default="images",
        help="Directory to save the plot(s) (default: images)"
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
        plot_temporal_3d(args.model, target_word=args.target, output_dir=args.output_dir, font_size=args.font_size)
    else:
        plot_batch(args.model_dir, args.target, args.output_dir, args.font_size)


if __name__ == "__main__":
    main()
