#!/usr/bin/env python3
"""
PPMI-based semantic change analysis (count-based, deterministic).

Alternative to TempRef Word2Vec. For each adjacent-period pair, builds two PPMI
matrices over pair-specific shared row and keyword vocabularies, then measures
how each reference word's similarity to the "interiority" target shifted.

No Procrustes alignment is needed because each adjacent pair shares the same
keyword columns. Use --global-vocab to select one vocabulary across all periods.

Usage (run from experiments/ppmi; data and result paths are relative to the repo root):
  python ppmi_semantic_change.py
  python ppmi_semantic_change.py --n-keywords 5000 --n-vocab 30000 --window 10
  python ppmi_semantic_change.py --global-vocab
  python ppmi_semantic_change.py --save-matrices
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from scipy import sparse

import jieba.posseg as pseg

PERIODS = ['mingqing', 'late_qing', 'republican', 'socialist', 'contemporary']
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_target_words(words_file: str) -> List[str]:
    if not os.path.exists(words_file):
        raise FileNotFoundError(f"Target words file not found: {words_file}")
    with open(words_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError(f"Target words file is empty: {words_file}")
    return words


def load_stopwords() -> set:
    try:
        from qhchina.helpers.texts import load_stopwords as _ls
        return _ls("zh_sim")
    except Exception:
        return set()


def iter_period_sentences(filepath: str, replacements: Dict[str, str]):
    """Yield tokenized sentences from a file with target word replacement."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if not words:
                continue
            yield [replacements.get(w, w) for w in words]


def compute_word_counts(data_dir: str, replacements: Dict[str, str]) -> Dict[str, Counter]:
    """Compute word counts per period (with target word replacement applied)."""
    counts = {}
    for period in PERIODS:
        filepath = os.path.join(data_dir, f"sentences_{period}.txt")
        if not os.path.exists(filepath):
            continue
        c = Counter()
        for sentence in tqdm(iter_period_sentences(filepath, replacements),
                              desc=f"counting {period}", leave=False):
            c.update(sentence)
        counts[period] = c
    return counts


def select_keywords(period_counts: Dict[str, Counter], periods: List[str],
                    n_keywords: int, min_freq: int, min_word_length: int,
                    stopwords: set) -> List[str]:
    """Select keywords meeting the frequency threshold in every requested period."""
    candidate_sets = []
    for period in periods:
        if period not in period_counts:
            return []
        s = set(w for w, c in period_counts[period].items()
                if c >= min_freq and len(w) >= min_word_length and w not in stopwords)
        candidate_sets.append(s)
    if not candidate_sets:
        return []
    shared = set.intersection(*candidate_sets)
    total_freq = {w: sum(period_counts[p].get(w, 0) for p in periods) for w in shared}
    ranked = sorted(shared, key=lambda word: (-total_freq[word], word))
    return ranked[:n_keywords]


def select_row_vocab(period_counts: Dict[str, Counter], periods: List[str],
                     target_token: str, n_vocab: int, min_count: int,
                     min_word_length: int, stopwords: set,
                     require_all_periods: bool) -> List[str]:
    """Select row words for the requested periods, always including the target."""
    candidate_sets = []
    for period in periods:
        if period not in period_counts:
            return []
        candidates = {
            w for w, count in period_counts[period].items()
            if count >= min_count and len(w) >= min_word_length and w not in stopwords
        }
        candidate_sets.append(candidates)

    if not candidate_sets:
        return []
    if require_all_periods:
        candidates = set.intersection(*candidate_sets)
    else:
        candidates = set.union(*candidate_sets)

    total_freq = {
        word: sum(period_counts[period].get(word, 0) for period in periods)
        for word in candidates
    }
    ranked = sorted(candidates, key=lambda word: (-total_freq[word], word))
    if target_token not in ranked:
        ranked.insert(0, target_token)
    return ranked[:n_vocab]


def compute_target_cooccurrences(filepath: str, replacements: Dict[str, str],
                                   target_token: str, window: int) -> Counter:
    """Count how often each word appears within `window` of the target token.

    This is separate from the PPMI matrix (which only stores co-occurrences
    with keyword columns). Here we scan the corpus and count co-occurrences
    with the `interiority` replacement token directly, for filtering/reporting.
    """
    cooc = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="target co-oc", leave=False):
            words = line.strip().split()
            if not words:
                continue
            # Apply replacements
            words = [replacements.get(w, w) for w in words]
            n = len(words)
            for i, tok in enumerate(words):
                if tok == target_token:
                    start = max(0, i - window)
                    end = min(n, i + window + 1)
                    for j in range(start, end):
                        if j == i:
                            continue
                        cooc[words[j]] += 1
    return cooc


def build_cooccurrence_matrix(filepath: str, replacements: Dict[str, str],
                               row_index: Dict[str, int],
                               col_index: Dict[str, int],
                               window: int) -> sparse.csr_matrix:
    """Build co-occurrence matrix: rows = row words (context), cols = col words (keywords).

    For each occurrence of a keyword, counts context words in its window.
    """
    n_rows = len(row_index)
    n_cols = len(col_index)
    matrix = sparse.lil_matrix((n_rows, n_cols), dtype=np.float64)
    col_set = set(col_index.keys())

    for sentence in tqdm(iter_period_sentences(filepath, replacements),
                        desc="co-occurrence", leave=False):
        n = len(sentence)
        for i, tok in enumerate(sentence):
            if tok in col_set:
                col_idx = col_index[tok]
                start = max(0, i - window)
                end = min(n, i + window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    other = sentence[j]
                    if other in row_index:
                        matrix[row_index[other], col_idx] += 1
    return matrix.tocsr()


def apply_ppmi(matrix: sparse.csr_matrix, smooth: float = 0.0) -> sparse.csr_matrix:
    """Apply standard PPMI normalization using the matrix's marginals."""
    matrix = matrix.tocsr().astype(np.float64)

    N = matrix.sum()
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    col_sums = np.array(matrix.sum(axis=0)).flatten()

    if N == 0:
        return matrix

    # Avoid log(0) for rows/cols that have no co-occurrences
    safe_row = np.where(row_sums > 0, row_sums, 1.0)
    safe_col = np.where(col_sums > 0, col_sums, 1.0)
    log_N = np.log(N + smooth)
    log_row = np.log(safe_row + smooth)
    log_col = np.log(safe_col + smooth)

    coo = matrix.tocoo()
    new_data = np.zeros_like(coo.data, dtype=np.float64)
    for k in range(len(coo.data)):
        i = coo.row[k]
        j = coo.col[k]
        val = coo.data[k]
        pmi = np.log(val + smooth) + log_N - log_row[i] - log_col[j]
        new_data[k] = max(0.0, pmi)

    ppmi = sparse.csr_matrix((new_data, (coo.row, coo.col)), shape=matrix.shape)
    ppmi.eliminate_zeros()
    return ppmi


def cosine_sim_sparse(vec_a: sparse.csr_matrix, mat_b: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity between a single row vector and each row of a matrix."""
    a = vec_a.toarray().flatten()
    a_norm = np.linalg.norm(a)
    if a_norm == 0:
        return np.zeros(mat_b.shape[0])
    B = mat_b.toarray()
    b_norms = np.linalg.norm(B, axis=1)
    denom = b_norms * a_norm
    denom[denom == 0] = 1.0
    return (B @ a) / denom


def cosine_sim_pair(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (na * nb))


def passes_filters(word: str, postag: Optional[str]) -> bool:
    if postag is not None:
        pos_result = pseg.lcut(word)
        if not pos_result or not re.match(postag, pos_result[0].flag):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PPMI-based semantic change analysis (deterministic, count-based)."
    )
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "segmented"),
                        help="Directory with per-period sentence files (default: data/segmented)")
    parser.add_argument("--words", default=os.path.join(ROOT, "data", "dictionaries", "interiority_words.txt"),
                        help="Path to interiority target words file")
    parser.add_argument("--replacement", default="interiority",
                        help="Replacement token for target words (default: interiority)")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "results", "ppmi"),
                        help="Output directory for CSV results (default: results/ppmi)")
    parser.add_argument("--n-keywords", type=int, default=5000,
                        help="Number of shared keyword columns (default: 5000)")
    parser.add_argument("--n-vocab", type=int, default=30000,
                        help="Max number of row words per period (default: 30000)")
    parser.add_argument("--window", type=int, default=10,
                        help="Context window size in words (default: 10)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Min word count in a period to be a row word (default: 5)")
    parser.add_argument("--min-keyword-freq", type=int, default=50,
                        help="Min frequency in each compared period to be a keyword "
                             "(default: 50)")
    parser.add_argument("--min-word-length", type=int, default=2,
                        help="Min word length (default: 2)")
    parser.add_argument("--min-cooc", type=int, default=5,
                        help="Min co-occurrence count with target for output (default: 5)")
    parser.add_argument("--postag", default="n.*",
                        help="POS tag regex filter for output words (default: n.*, 'none' to disable)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Top N words per transition in output (default: 100)")
    parser.add_argument("--save-matrices", action="store_true",
                        help="Save PPMI matrices (.npz) and vocab/keyword lists for later inspection")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Additive smoothing for PPMI (default: 0.0)")
    parser.add_argument("--global-vocab", action="store_true",
                        help="Select one keyword and row vocabulary across all periods "
                             "instead of selecting them separately for each adjacent pair")
    args = parser.parse_args()

    postag = None if args.postag.lower() == 'none' else args.postag

    # Load target words and build replacements
    target_words = load_target_words(args.words)
    replacements = {word: args.replacement for word in target_words}
    print(f"Loaded {len(target_words)} target words, replacement: '{args.replacement}'")

    stopwords = load_stopwords()
    print(f"Loaded {len(stopwords)} stopwords")

    # Compute per-period word counts
    print("\nComputing per-period word counts...")
    period_counts = compute_word_counts(args.data_dir, replacements)
    for period in PERIODS:
        if period in period_counts:
            print(f"  {period}: {len(period_counts[period]):,} unique words, "
                  f"{sum(period_counts[period].values()):,} tokens")

    available_periods = [period for period in PERIODS if period in period_counts]
    vocab_mode = "global" if args.global_vocab else "adjacent-pair"
    print(f"\nVocabulary mode: {vocab_mode}")

    global_keywords = None
    global_row_vocab = None
    if args.global_vocab:
        print(f"Selecting up to {args.n_keywords} keywords shared by all periods...")
        global_keywords = select_keywords(
            period_counts, available_periods, args.n_keywords,
            args.min_keyword_freq, args.min_word_length, stopwords
        )
        global_row_vocab = select_row_vocab(
            period_counts, available_periods, args.replacement, args.n_vocab,
            args.min_count, args.min_word_length, stopwords,
            require_all_periods=False,
        )
        print(f"  Selected {len(global_keywords)} keywords and "
              f"{len(global_row_vocab)} row words")
        if not global_keywords:
            print("Error: No globally shared keywords found. "
                  "Try lowering --min-keyword-freq.")
            sys.exit(1)

    # Target co-occurrence counts do not depend on the matrix vocabulary.
    target_cooc = {}  # period -> Counter of co-occurrences with the interiority token
    print("\nComputing target co-occurrences...")
    for period in available_periods:
        filepath = os.path.join(args.data_dir, f"sentences_{period}.txt")
        print(f"  {period}...")
        target_cooc[period] = compute_target_cooccurrences(
            filepath, replacements, args.replacement, args.window
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print("\nComputing semantic change scores per transition...")

    for i in range(len(PERIODS) - 1):
        from_period = PERIODS[i]
        to_period = PERIODS[i + 1]
        transition = f"{from_period}_to_{to_period}"

        if from_period not in period_counts or to_period not in period_counts:
            print(f"\nSkipping {transition}: missing corpus")
            continue

        pair_periods = [from_period, to_period]
        if args.global_vocab:
            keywords = global_keywords
            row_vocab = global_row_vocab
        else:
            keywords = select_keywords(
                period_counts, pair_periods, args.n_keywords,
                args.min_keyword_freq, args.min_word_length, stopwords
            )
            row_vocab = select_row_vocab(
                period_counts, pair_periods, args.replacement, args.n_vocab,
                args.min_count, args.min_word_length, stopwords,
                require_all_periods=True,
            )

        print(f"\n{transition}:")
        print(f"  Selected {len(keywords)} keywords and {len(row_vocab)} row words")
        if not keywords:
            print("  Skipping: no shared keywords. Try lowering --min-keyword-freq.")
            continue
        if args.replacement not in row_vocab:
            print(f"  Skipping: target token '{args.replacement}' is unavailable.")
            continue

        keyword_index = {word: idx for idx, word in enumerate(keywords)}
        row_index = {word: idx for idx, word in enumerate(row_vocab)}
        target_idx = row_index[args.replacement]

        ppmi_matrices = {}
        for period in pair_periods:
            filepath = os.path.join(args.data_dir, f"sentences_{period}.txt")
            cooc = build_cooccurrence_matrix(
                filepath, replacements, row_index, keyword_index, args.window
            )
            ppmi = apply_ppmi(cooc, smooth=args.smoothing)
            ppmi_matrices[period] = ppmi
            print(f"  {period}: PPMI matrix {ppmi.shape}, "
                  f"{ppmi.nnz:,} non-zero entries")

        if args.save_matrices:
            matrices_dir = os.path.join(args.output_dir, "matrices", transition)
            os.makedirs(matrices_dir, exist_ok=True)
            for period, mat in ppmi_matrices.items():
                sparse.save_npz(os.path.join(matrices_dir, f"ppmi_{period}.npz"), mat)
            with open(os.path.join(matrices_dir, "row_vocab.txt"),
                      'w', encoding='utf-8') as f:
                for word in row_vocab:
                    f.write(word + "\n")
            with open(os.path.join(matrices_dir, "keywords.txt"),
                      'w', encoding='utf-8') as f:
                for word in keywords:
                    f.write(word + "\n")
            print(f"  Saved matrices and vocab to {matrices_dir}/")

        mat_from = ppmi_matrices[from_period]
        mat_to = ppmi_matrices[to_period]

        target_vec_from = mat_from[target_idx].toarray().flatten()
        target_vec_to = mat_to[target_idx].toarray().flatten()

        # For each row word present in both periods, compute:
        #   change(w) = cos(w_to, target_to) - cos(w_from, target_from)
        rows = []
        for w_idx, word in enumerate(row_vocab):
            if word == args.replacement:
                continue

            from_counts_w = period_counts.get(from_period, Counter()).get(word, 0)
            to_counts_w = period_counts.get(to_period, Counter()).get(word, 0)
            if from_counts_w < args.min_count or to_counts_w < args.min_count:
                continue

            if not passes_filters(word, postag):
                continue

            w_vec_from = mat_from[w_idx].toarray().flatten()
            w_vec_to = mat_to[w_idx].toarray().flatten()

            # Skip words with no signal in either period
            if np.all(w_vec_from == 0) and np.all(w_vec_to == 0):
                continue

            sim_from = cosine_sim_pair(w_vec_from, target_vec_from)
            sim_to = cosine_sim_pair(w_vec_to, target_vec_to)
            change = sim_to - sim_from

            # Co-occurrence count with target in each period (computed from corpus)
            cooc_pre = target_cooc.get(from_period, Counter()).get(word, 0)
            cooc_post = target_cooc.get(to_period, Counter()).get(word, 0)
            if cooc_post < args.min_cooc:
                continue

            rows.append({
                'word': word,
                'change_score': change,
                'sim_pre': sim_from,
                'sim_post': sim_to,
                'count_pre': from_counts_w,
                'count_post': to_counts_w,
                'cooc_pre': cooc_pre,
                'cooc_post': cooc_post,
            })

        rows.sort(key=lambda x: x['change_score'], reverse=True)
        rows = rows[:args.top_n]

        csv_path = os.path.join(args.output_dir, f"semantic_changes_{transition}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "word", "change_score", "sim_pre", "sim_post",
                "count_pre", "count_post", "cooc_pre", "cooc_post"
            ])
            for rank, row in enumerate(rows, 1):
                writer.writerow([
                    rank, row['word'],
                    f"{row['change_score']:.6f}",
                    f"{row['sim_pre']:.6f}",
                    f"{row['sim_post']:.6f}",
                    row['count_pre'], row['count_post'],
                    row['cooc_pre'], row['cooc_post'],
                ])
        print(f"  Wrote {csv_path} ({len(rows)} words)")

    print(f"\nPPMI analysis complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
