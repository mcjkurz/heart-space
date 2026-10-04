#!/usr/bin/env python3
"""
Chinese text segmentation using jieba with multiprocessing support.

Segments normalized Chinese texts into sentences of words, grouped by historical
period. Outputs per-period text files (one sentence per line, space-separated words).

Usage (run from project root):
  python scripts/segment_texts.py
  python scripts/segment_texts.py --input-dir data/texts_normalized --output-dir data/segmented
  python scripts/segment_texts.py --workers 4
"""

import argparse
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm.auto import tqdm


PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']

_worker_initialized = False


def _init_worker(dict_path):
    """Initialize jieba in each worker process (called once per worker)."""
    global _worker_initialized
    if not _worker_initialized:
        import jieba
        if dict_path and os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
        _worker_initialized = True


def _segment_text(args):
    """Segment a single text (text content passed from main process)."""
    text, min_sentence_length = args
    import jieba.posseg as pseg

    sentences = []
    for line in text.split("\n"):
        parts = re.split(r'[。！？；]', line)
        for s in parts:
            words = [word for word, tag in pseg.lcut(s) if tag != "x"]
            if len(words) >= min_sentence_length:
                sentences.append(words)

    seen_words = set()
    for sentence in sentences:
        seen_words.update(sentence)

    return sentences, seen_words


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


def build_dictionary(dict_path, target_words, output_path):
    """Build jieba user dictionary from classical Chinese dictionary + target words."""
    dict_words = []
    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            dict_words = [line.strip() for line in f if line.strip()]

    dict_words.extend(target_words)
    dict_words = list(set(dict_words))
    print(f"Built dictionary of {len(dict_words)} words (dictionary + target words)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in dict_words:
            f.write(f"{word} 100000 nr\n")

    return output_path


def segment_period(input_dir, period, executor, min_sentence_length=5):
    """Segment all .txt files in a period directory using a shared executor."""
    period_dir = os.path.join(input_dir, period)
    if not os.path.isdir(period_dir):
        print(f"  Warning: directory not found: {period_dir}")
        return [], Counter()

    txt_files = sorted(f for f in os.listdir(period_dir) if f.endswith(".txt"))
    print(f"  {period}: {len(txt_files)} files")

    texts = []
    for txt_file in txt_files:
        filepath = os.path.join(period_dir, txt_file)
        with open(filepath, "r", encoding="utf-8") as f:
            texts.append(f.read())

    all_sentences = []
    in_how_many_texts = Counter()

    tasks = [(text, min_sentence_length) for text in texts]
    futures = [executor.submit(_segment_text, task) for task in tasks]

    for future in tqdm(futures, total=len(futures), desc=f"  {period}", leave=False):
        sentences, seen_words = future.result()
        all_sentences.extend(sentences)
        in_how_many_texts.update({w: 1 for w in seen_words})

    return all_sentences, in_how_many_texts


def main():
    parser = argparse.ArgumentParser(
        description="Segment normalized Chinese texts into sentences using jieba."
    )
    parser.add_argument(
        "--input-dir", default="data/texts_normalized",
        help="Directory containing period subdirectories of normalized texts (default: data/texts_normalized)"
    )
    parser.add_argument(
        "--output-dir", default="data/segmented",
        help="Output directory for segmented data (default: data/segmented)"
    )
    parser.add_argument(
        "--dict", default="data/dictionaries/古代汉语词典.txt",
        help="Path to classical Chinese dictionary file"
    )
    parser.add_argument(
        "--words", default="data/dictionaries/interiority_words.txt",
        help="Path to target words file"
    )
    parser.add_argument(
        "--min-length", type=int, default=5,
        help="Minimum sentence length in words (default: 5)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling sentences (default: 42)"
    )
    args = parser.parse_args()

    target_words = load_target_words(args.words)
    print(f"Loaded {len(target_words)} target words")

    dict_output = os.path.join(os.path.dirname(args.dict) or "data/dictionaries", "final_dict.txt")
    dict_path = build_dictionary(args.dict, target_words, dict_output)

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    total_sentences = 0
    total_text_counts = Counter()

    print(f"\nSegmenting texts from {args.input_dir} (workers={args.workers})...")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(dict_path,)) as executor:
        for period in PERIODS:
            sentences, text_counts = segment_period(
                args.input_dir, period, executor,
                min_sentence_length=args.min_length
            )
            total_text_counts.update(text_counts)

            random.shuffle(sentences)

            txt_path = os.path.join(args.output_dir, f"sentences_{period}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for sentence in sentences:
                    f.write(" ".join(sentence) + "\n")

            print(f"  {period}: {len(sentences)} sentences -> {txt_path}")
            total_sentences += len(sentences)

    print(f"\nDone. Total: {total_sentences} sentences across {len(PERIODS)} periods.")


if __name__ == "__main__":
    main()
