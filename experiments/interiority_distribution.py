#!/usr/bin/env python3
"""
Visualize the distribution of interiority words throughout a novel.

Creates a "barcode" style visualization where each vertical line marks
an occurrence of an interiority word. The horizontal axis represents
the position in the novel text.

Usage (run from project root):
  python experiments/interiority_distribution.py
"""

import matplotlib.pyplot as plt
import numpy as np


def load_interiority_words(filepath: str) -> list[str]:
    """Load interiority words from a text file (one word per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def find_word_positions(text: str, words: list[str]) -> list[float]:
    """
    Find all positions of interiority words in the text.
    
    Returns normalized positions (0.0 to 1.0) relative to text length.
    """
    positions = []
    text_length = len(text)
    
    for word in words:
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            normalized_pos = pos / text_length
            positions.append(normalized_pos)
            start = pos + 1
    
    return sorted(positions)


def create_barcode_visualization(
    positions: list[float],
    output_path: str,
    width_px: int = 1024,
    height_px: int = 150,
    dpi: int = 300
):
    """
    Create a barcode-style visualization of word positions.
    
    Each position is marked with a vertical line spanning the full height.
    """
    fig_width = width_px / dpi
    fig_height = height_px / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    for pos in positions:
        ax.axvline(x=pos, color='steelblue', linewidth=0.3, alpha=0.7)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], fontsize=3, color='black')
    ax.tick_params(axis='x', length=1.5, width=0.5, pad=1, colors='black')
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"Saved visualization to: {output_path}")


def main():
    novel_path = "data/cuotuo_suiyue.txt"
    words_path = "data/dictionaries/interiority_words.txt"
    output_path = "images/interiority_distribution_cuotuo_suiyue.png"
    
    print("Loading interiority words...")
    words = load_interiority_words(words_path)
    print(f"Loaded {len(words)} interiority words")
    
    print("Loading novel text...")
    with open(novel_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Novel length: {len(text)} characters")
    
    print("Finding word positions...")
    positions = find_word_positions(text, words)
    print(f"Found {len(positions)} occurrences of interiority words")
    
    print("Creating visualization...")
    create_barcode_visualization(
        positions,
        output_path,
        width_px=1024,
        height_px=150,
        dpi=300
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
