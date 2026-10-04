"""Visualization functions for semantic change analysis.

These functions can be called programmatically (passing a model object)
or from the command line via draw_heatmap.py / visualize_interiority_3d.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from qhchina.analytics import TempRefWord2Vec

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for heatmap")


def _load_model(model_or_path: Union[TempRefWord2Vec, str]) -> TempRefWord2Vec:
    """Accept either a model object or a path and return a loaded model."""
    if isinstance(model_or_path, str):
        print(f"Loading model from {model_or_path}...")
        return TempRefWord2Vec.load(model_or_path)
    return model_or_path


def plot_semantic_change_distribution(
    changes: Dict[str, List[Tuple[str, float]]],
    target_word: str,
    bin_size: float = 0.05,
    save_dir: str = "."
) -> None:
    """Plot histogram distributions of semantic changes for each transition in a 2x2 subplot."""
    title_mapping = {
        'mingqing_to_late_qing': 'Ming-Qing \u2192 Late Qing',
        'late_qing_to_republican': 'Late Qing \u2192 Republican',
        'republican_to_socialist': 'Republican \u2192 Socialist',
        'socialist_to_contemporary': 'Socialist \u2192 Contemporary'
    }

    def get_readable_title(transition_key):
        if transition_key in title_mapping:
            return title_mapping[transition_key]
        if '_to_' in transition_key:
            parts = transition_key.split('_to_')
            if len(parts) == 2:
                from_period = parts[0].replace('_', '-').title()
                to_period = parts[1].replace('_', '-').title()
                return f"{from_period} \u2192 {to_period}"
        return transition_key.replace('_to_', ' \u2192 ').replace('_', ' ').title()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = {
        'negative': '#D32F2F',
        'positive': '#1976D2',
        'zero_line': '#424242',
        'text_bg': '#FFF3E0'
    }

    for idx, (transition, word_changes) in enumerate(changes.items()):
        if idx >= 4:
            break

        ax = axes[idx]

        if not word_changes:
            ax.text(0.5, 0.5, f'No data for\n{get_readable_title(transition)}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(get_readable_title(transition), fontsize=14, fontweight='bold')
            continue

        change_values = [change for word, change in word_changes]

        x_min, x_max = min(change_values), max(change_values)

        if x_min > 0:
            x_min = min(x_min, -bin_size/2)
        if x_max < 0:
            x_max = max(x_max, bin_size/2)

        n_bins_neg = int(np.ceil(-x_min / bin_size))
        n_bins_pos = int(np.ceil(x_max / bin_size))

        bins_neg = np.linspace(-n_bins_neg * bin_size, 0, n_bins_neg + 1)
        bins_pos = np.linspace(0, n_bins_pos * bin_size, n_bins_pos + 1)
        bins = np.concatenate([bins_neg[:-1], bins_pos])

        print(f"\nPlotting distribution for {transition}")
        print(f"  Vocab size: {len(change_values):,}, range: [{min(change_values):.4f}, {max(change_values):.4f}]")
        print(f"  Mean: {np.mean(change_values):.4f}, std: {np.std(change_values):.4f}")

        counts, bin_edges, patches = ax.hist(change_values, bins=bins, alpha=0.8,
                                           edgecolor='white', linewidth=0.5)

        for i, (count, left_edge, right_edge) in enumerate(zip(counts, bin_edges[:-1], bin_edges[1:])):
            bin_center = (left_edge + right_edge) / 2
            if bin_center < 0:
                patches[i].set_color(colors['negative'])
            else:
                patches[i].set_color(colors['positive'])

        ax.axvline(x=0, color=colors['zero_line'], linestyle='--', linewidth=2, alpha=0.9, zorder=10)

        stats_text = (
            f"Stats:\n"
            f"Total: {len(change_values):,}\n"
            f"Mean: {np.mean(change_values):.3f}\n"
            f"Std: {np.std(change_values):.3f}\n\n"
            f"Closer (>0): {sum(1 for x in change_values if x > 0):,}\n"
            f"Away (<0): {sum(1 for x in change_values if x < 0):,}"
        )

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor=colors['text_bg'], alpha=0.9),
               fontsize=11, fontfamily='monospace')

        ax.set_title(get_readable_title(transition), fontsize=14, fontweight='bold')
        ax.set_xlabel('Semantic Change Score', fontsize=11)
        ax.set_ylabel('Number of Words', fontsize=11)
        ax.grid(True, alpha=0.3)

        x_abs_max = max(abs(x_min), abs(x_max))
        x_margin = x_abs_max * 0.05
        ax.set_xlim(-x_abs_max - x_margin, x_abs_max + x_margin)

        symmetric_range = 2 * x_abs_max
        if symmetric_range <= 0.5:
            tick_step = 0.05
        elif symmetric_range <= 1.0:
            tick_step = 0.1
        elif symmetric_range <= 2.0:
            tick_step = 0.2
        else:
            tick_step = 0.5

        max_tick = np.floor(x_abs_max / tick_step) * tick_step
        ax.set_xticks(np.arange(-max_tick, max_tick + tick_step, tick_step))

    for idx in range(len(changes), 4):
        axes[idx].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color=colors['zero_line'], linestyle='--', linewidth=2, label='Zero line'),
        Patch(facecolor=colors['negative'], alpha=0.8, label='Moving away (negative)'),
        Patch(facecolor=colors['positive'], alpha=0.8, label='Moving closer (positive)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, fontsize=14, frameon=True, fancybox=True, shadow=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    os.makedirs(save_dir, exist_ok=True)
    filename = f"semantic_change_distribution_{target_word}_combined.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved combined plot: {filepath}")


def plot_temporal_similarity_heatmap(
    model_or_path: Union[TempRefWord2Vec, str],
    target_word: str,
    save_dir: str = "."
) -> None:
    """Create a heatmap showing cosine similarities between temporal variants.

    Args:
        model_or_path: A TempRefWord2Vec model object, or a path to a .npy model file.
        target_word: The target word whose temporal variants to compare.
        save_dir: Directory to save the heatmap image.
    """
    model = _load_model(model_or_path)

    periods = ['Ming-Qing', 'Late Qing', 'Republican', 'Socialist', 'Contemporary']
    model_labels = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']

    periods_display = periods[::-1]

    n_periods = len(periods)
    similarity_matrix = np.zeros((n_periods, n_periods))

    print(f"\nCreating temporal similarity heatmap for '{target_word}'...")

    for i, period_i in enumerate(model_labels):
        for j, period_j in enumerate(model_labels):
            temporal_variant_i = f"{target_word}_{period_i}"
            temporal_variant_j = f"{target_word}_{period_j}"

            try:
                if i == j:
                    similarity = 1.0
                else:
                    similarity = model.similarity(temporal_variant_i, temporal_variant_j)

                similarity_matrix[i, j] = similarity
                print(f"  {periods[i]} <-> {periods[j]}: {similarity:.4f}")

            except KeyError as e:
                print(f"  Warning: Temporal variant not found: {e}")
                similarity_matrix[i, j] = np.nan
            except Exception as e:
                print(f"  Error: {periods[i]} <-> {periods[j]}: {e}")
                similarity_matrix[i, j] = np.nan

    plt.figure(figsize=(10, 8))

    similarity_matrix_display = np.flipud(similarity_matrix)
    mask = np.isnan(similarity_matrix_display)

    if HAS_SEABORN:
        sns.heatmap(similarity_matrix_display,
                    xticklabels=periods,
                    yticklabels=periods_display,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    vmin=0.0,
                    vmax=1.0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'label': 'Cosine Similarity'},
                    mask=mask,
                    annot_kws={'size': 14})

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    else:
        im = plt.imshow(similarity_matrix_display, cmap='YlOrRd', vmin=0.0, vmax=1.0, aspect='equal')
        plt.colorbar(im, label='Cosine Similarity')

        for i in range(n_periods):
            for j in range(n_periods):
                if not np.isnan(similarity_matrix_display[i, j]):
                    plt.text(j, i, f'{similarity_matrix_display[i, j]:.3f}',
                           ha='center', va='center', fontsize=14,
                           color='black' if similarity_matrix_display[i, j] < 0.7 else 'white')

        plt.xticks(range(n_periods), periods, rotation=0)
        plt.yticks(range(n_periods), periods_display, rotation=0)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"temporal_similarity_heatmap_{target_word}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved similarity heatmap: {filepath}")

    valid_similarities = similarity_matrix[~np.isnan(similarity_matrix)]
    if len(valid_similarities) > 0:
        print(f"\nSimilarity matrix stats:")
        print(f"  Mean: {np.mean(valid_similarities):.4f}")
        print(f"  Std: {np.std(valid_similarities):.4f}")
        print(f"  Min: {np.min(valid_similarities):.4f}")
        print(f"  Max: {np.max(valid_similarities):.4f}")

        off_diagonal_mask = ~np.eye(n_periods, dtype=bool) & ~np.isnan(similarity_matrix)
        if np.any(off_diagonal_mask):
            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix * off_diagonal_mask.astype(float)), similarity_matrix.shape)
            min_sim_idx = np.unravel_index(np.argmin(similarity_matrix + (~off_diagonal_mask).astype(float) * 999), similarity_matrix.shape)

            print(f"  Most similar: {periods[max_sim_idx[0]]} <-> {periods[max_sim_idx[1]]} ({similarity_matrix[max_sim_idx]:.4f})")
            print(f"  Least similar: {periods[min_sim_idx[0]]} <-> {periods[min_sim_idx[1]]} ({similarity_matrix[min_sim_idx]:.4f})")
