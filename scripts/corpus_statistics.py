#!/usr/bin/env python3
"""
Corpus statistics across five historical periods.

Prints character counts for raw texts and normalized texts, and word/sentence
statistics for segmented data. Also generates a corpus_statistics.txt file
containing all statistics plus a full text listing.

Usage (run from project root):
  python scripts/corpus_statistics.py
  python scripts/corpus_statistics.py --texts-dir data/texts --normalized-dir data/texts_normalized --segmented-dir data/segmented
"""

import argparse
import os
from collections import Counter
from pathlib import Path

from qhchina.utils import LineSentenceFile

PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']
PERIOD_LABELS = {
    'mingqing': 'Ming-Qing',
    'late_qing': 'Late Qing',
    'republican': 'Republican',
    'socialist': 'Socialist',
    'contemporary': 'Contemporary',
}


def count_chars_in_dir(directory):
    """Return (total_chars, num_files) for all .txt files in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return None, 0
    total = 0
    n_files = 0
    for f in sorted(d.glob("*.txt")):
        with open(f, "r", encoding="utf-8") as fh:
            total += len(fh.read())
        n_files += 1
    return total, n_files


def get_text_entries(texts_dir):
    """Get author《title》entries grouped by period."""
    texts_dir = Path(texts_dir)
    period_entries = {}
    for period in PERIODS:
        period_dir = texts_dir / period
        if not period_dir.is_dir():
            period_entries[period] = []
            continue
        entries = []
        for f in sorted(period_dir.glob("*.txt")):
            stem = f.stem
            if "_" in stem:
                author, title = stem.split("_", 1)
            else:
                author, title = "", stem
            entries.append(f"{author}《{title}》")
        period_entries[period] = entries
    return period_entries


def compute_period_stats(filepath):
    """Compute all stats for a period in a single pass (streaming)."""
    n_sentences = 0
    word_counts = Counter()
    total_words = 0
    total_chars = 0
    
    for sent in LineSentenceFile(filepath):
        n_sentences += 1
        total_words += len(sent)
        for w in sent:
            word_counts[w] += 1
            total_chars += len(w)
    
    vocab_size = len(word_counts)
    avg_sent_len = total_words / n_sentences if n_sentences else 0
    
    return {
        'sentences': n_sentences,
        'total_words': total_words,
        'unique_words': vocab_size,
        'total_chars': total_chars,
        'avg_sent_len': avg_sent_len,
        'word_counts': word_counts,
    }


def fmt(n):
    """Format an integer with thousands separators."""
    return f"{n:,}"


def load_interiority_words(words_file):
    """Load interiority words from file."""
    if not os.path.exists(words_file):
        return []
    with open(words_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Corpus statistics by period.")
    parser.add_argument("--texts-dir", default="data/texts",
                        help="Directory with raw texts (default: data/texts)")
    parser.add_argument("--normalized-dir", default="data/texts_normalized",
                        help="Directory with normalized texts (default: data/texts_normalized)")
    parser.add_argument("--segmented-dir", default="data/segmented",
                        help="Directory with segmented data (default: data/segmented)")
    parser.add_argument("--words", default="data/dictionaries/interiority_words.txt",
                        help="Path to interiority words file (default: data/dictionaries/interiority_words.txt)")
    parser.add_argument("--output", default="data/texts/corpus_statistics.txt",
                        help="Output path for statistics file (default: data/texts/corpus_statistics.txt)")
    args = parser.parse_args()

    lines = []

    period_stats = {}
    for period in PERIODS:
        filepath = os.path.join(args.segmented_dir, f"sentences_{period}.txt")
        if os.path.exists(filepath):
            print(f"Loading {period}...")
            period_stats[period] = compute_period_stats(filepath)

    # ── Raw texts ──────────────────────────────────────────────────────
    lines.append("=" * 86)
    lines.append("RAW TEXTS  (data/texts)")
    lines.append("=" * 86)
    lines.append(f"{'Period':<16} {'Files':>8} {'Characters':>14}")
    lines.append("-" * 42)
    grand_chars = 0
    grand_files = 0
    for period in PERIODS:
        d = os.path.join(args.texts_dir, period)
        chars, n_files = count_chars_in_dir(d)
        if chars is None:
            lines.append(f"{PERIOD_LABELS[period]:<16} {'(missing)':>8}")
            continue
        grand_chars += chars
        grand_files += n_files
        lines.append(f"{PERIOD_LABELS[period]:<16} {n_files:>8} {fmt(chars):>14}")
    lines.append("-" * 42)
    lines.append(f"{'TOTAL':<16} {grand_files:>8} {fmt(grand_chars):>14}")

    # ── Normalized texts ───────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 86)
    lines.append("NORMALIZED TEXTS  (data/texts_normalized)")
    lines.append("=" * 86)
    lines.append(f"{'Period':<16} {'Files':>8} {'Characters':>14}")
    lines.append("-" * 42)
    grand_chars = 0
    grand_files = 0
    has_normalized = False
    for period in PERIODS:
        d = os.path.join(args.normalized_dir, period)
        chars, n_files = count_chars_in_dir(d)
        if chars is None:
            lines.append(f"{PERIOD_LABELS[period]:<16} {'(missing)':>8}")
            continue
        has_normalized = True
        grand_chars += chars
        grand_files += n_files
        lines.append(f"{PERIOD_LABELS[period]:<16} {n_files:>8} {fmt(chars):>14}")
    if has_normalized:
        lines.append("-" * 42)
        lines.append(f"{'TOTAL':<16} {grand_files:>8} {fmt(grand_chars):>14}")
    else:
        lines.append("  (no normalized texts found — run normalize_texts.py first)")

    # ── Segmented data ─────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 86)
    lines.append("SEGMENTED DATA  (data/segmented)")
    lines.append("=" * 86)
    if not period_stats:
        lines.append("  (no segmented files found — run segment_texts.py first)")
    else:
        lines.append(f"{'Period':<16} {'Sentences':>12} {'Total Words':>14} "
                     f"{'Unique Words':>14} {'Total Chars':>14} {'Avg Sent Len':>14}")
        lines.append("-" * 86)
        totals = {'sentences': 0, 'total_words': 0, 'total_chars': 0}
        all_word_counts = Counter()
        for period in PERIODS:
            if period not in period_stats:
                continue
            stats = period_stats[period]
            totals['sentences'] += stats['sentences']
            totals['total_words'] += stats['total_words']
            totals['total_chars'] += stats['total_chars']
            all_word_counts.update(stats['word_counts'])
            lines.append(f"{PERIOD_LABELS[period]:<16} {fmt(stats['sentences']):>12} "
                         f"{fmt(stats['total_words']):>14} {fmt(stats['unique_words']):>14} "
                         f"{fmt(stats['total_chars']):>14} {stats['avg_sent_len']:>14.1f}")
        lines.append("-" * 86)
        global_vocab = len(all_word_counts)
        global_avg = (totals['total_words'] / totals['sentences']
                      if totals['sentences'] else 0)
        lines.append(f"{'TOTAL':<16} {fmt(totals['sentences']):>12} "
                     f"{fmt(totals['total_words']):>14} {fmt(global_vocab):>14} "
                     f"{fmt(totals['total_chars']):>14} {global_avg:>14.1f}")

    # ── Interiority word frequencies ────────────────────────────────────
    interiority_words = load_interiority_words(args.words)
    if interiority_words and period_stats:
        lines.append("")
        lines.append("=" * 120)
        lines.append("INTERIORITY WORD FREQUENCIES  (absolute count / relative frequency per 10,000 words)")
        lines.append("=" * 120)
        
        available_periods = [p for p in PERIODS if p in period_stats]
        header = f"{'Word':<12}"
        for period in available_periods:
            header += f" {PERIOD_LABELS[period]:>20}"
        lines.append(header)
        lines.append("-" * 120)
        
        for word in interiority_words:
            row = f"{word:<12}"
            for period in available_periods:
                count = period_stats[period]['word_counts'].get(word, 0)
                total = period_stats[period]['total_words']
                rel_freq = (count / total * 10000) if total > 0 else 0
                row += f" {count:>8} ({rel_freq:>6.2f})"
            lines.append(row)
        
        lines.append("-" * 120)
        row = f"{'TOTAL':<12}"
        for period in available_periods:
            period_count = sum(period_stats[period]['word_counts'].get(w, 0) for w in interiority_words)
            total = period_stats[period]['total_words']
            rel_freq = (period_count / total * 10000) if total > 0 else 0
            row += f" {period_count:>8} ({rel_freq:>6.2f})"
        lines.append(row)

    # ── Text listing ───────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 86)
    lines.append("TEXT LISTING")
    lines.append("=" * 86)
    
    period_entries = get_text_entries(args.texts_dir)
    for period in PERIODS:
        entries = period_entries[period]
        if not entries:
            continue
        lines.append("")
        lines.append(f"[{PERIOD_LABELS[period]}] ({len(entries)} texts)")
        lines.append("")
        for entry in entries:
            lines.append(entry)
    
    lines.append("")

    # Print to console
    for line in lines:
        print(line)

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Statistics written to: {output_path}")


if __name__ == "__main__":
    main()
