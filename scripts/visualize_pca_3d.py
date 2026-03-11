#!/usr/bin/env python3
"""
3D PCA visualization of temporal word variants.

Loads a TempRefWord2Vec model, extracts vectors for temporal variants of a
target word, reduces to 3D with PCA, and plots the semantic trajectory.

Usage (run from project root):
  python scripts/visualize_pca_3d.py
  python scripts/visualize_pca_3d.py --model models/tempref_stable_words.npy --target 声响
"""

import argparse
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


def plot_temporal_3d(model_path, target_word="interiority", output_dir="images"):
    """Load model, extract temporal variants, PCA to 3D, and save a plot."""
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
                fontsize=11, fontweight='bold', ha='center', alpha=0.8)

    if len(vectors_3d) > 1:
        ax.plot(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
                'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel(f'PC1 ({explained[0]:.1%} variance)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%} variance)', fontweight='bold')
    ax.set_zlabel(f'PC3 ({explained[2]:.1%} variance)', fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[i], markersize=10,
                   label=PERIOD_LEGEND_LABELS.get(label, label))
        for i, label in enumerate(valid_labels)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=11)

    plt.tight_layout(pad=3.0)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'temporal_{target_word}_3d_pca.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"\nPlot saved: {output_file}")

    plt.show()

    print(f"\nTarget: {target_word}")
    print(f"Variants: {len(valid_variants)}, PCA total variance: {explained.sum():.3f}")
    if len(vectors_3d) > 1:
        print("\nConsecutive distances (3D PCA space):")
        for i in range(len(vectors_3d) - 1):
            dist = np.linalg.norm(vectors_3d[i+1] - vectors_3d[i])
            print(f"  {valid_labels[i]} -> {valid_labels[i+1]}: {dist:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="3D PCA visualization of temporal word variants."
    )
    parser.add_argument(
        "--model", default="models/tempref_interiority_w2v.npy",
        help="Path to TempRefWord2Vec model file"
    )
    parser.add_argument(
        "--target", default="interiority",
        help="Target word to visualize (default: interiority)"
    )
    parser.add_argument(
        "--output-dir", default="images",
        help="Directory to save the plot (default: images)"
    )
    args = parser.parse_args()

    plot_temporal_3d(args.model, target_word=args.target, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
