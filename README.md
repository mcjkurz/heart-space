# Interiority Semantic Change Analysis

Computational analysis of semantic change in Chinese "interiority" words (心里, 心中, 内心, etc.) across five historical periods using temporal referencing Word2Vec (TempRef).

## Periods


| Period       | Label          | Approximate Dates |
| ------------ | -------------- | ----------------- |
| Ming-Qing    | `mingqing`     | 1368 -- 1860      |
| Late Qing    | `late_qing`    | 1860 -- 1911      |
| Republican   | `republican`   | 1911 -- 1949      |
| Socialist    | `socialist`    | 1949 -- 1976      |
| Contemporary | `contemporary` | 1976 -- present   |


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
data/
  texts/                 Raw text corpora by period
  texts_normalized/      Simplified/normalized texts (Step 1)
  segmented/             Per-period sentence files (Step 2)
  dictionaries/          Jieba dictionaries and word lists
models/                  Trained models (.npy)
results/                 Analysis output CSVs
images/                  Generated plots
scripts/                 Pipeline and analysis scripts
```

## Workflow

### Step 1: Normalize texts

```bash
python scripts/normalize_texts.py
```

Converts traditional to simplified Chinese, removes noise, normalizes punctuation.

### Step 2: Segment texts

```bash
python scripts/segment_texts.py
```

Segments normalized texts into words using dictionary-enhanced jieba. Produces `data/segmented/sentences_{period}.txt`.

### Step 3: Train models

```bash
# Single model
python scripts/train_tempref.py --output-dir models/real/

# 100 models with parallel training
python scripts/train_tempref.py --trials 100 --processes 4 --output-dir models/real/

# Null distribution (shuffled periods)
python scripts/train_tempref.py --trials 100 --processes 4 --permute-periods --output-dir models/null/
```

Model naming: `model_e{epochs}_s{seed}.npy` (or `_null.npy` for permuted).

### Step 4: Analyze semantic changes

```bash
# Single model
python scripts/semantic_change.py --model models/real/model_e3_s42.npy

# Multi-model ensemble with null distribution
python scripts/semantic_change.py --model-dir models/real/ --null-dir models/null/ --output-dir results/
```

Produces per-transition CSVs (e.g., `semantic_changes_mingqing_to_late_qing.csv`) with columns: `word`, `mean_change`, `std_change`, `z_score`, `p_value`, etc.

## Additional Scripts


| Script                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `train_period_models.py`   | Train separate Word2Vec models per period         |
| `analyze_period_models.py` | Analyze per-period models (multi-trial averaging) |
| `find_stable_words.py`     | Find semantically stable words across periods     |
| `epoch_validation.py`      | Compare models at different epoch counts          |
| `ensemble_stability.py`    | Analyze how ensemble size affects reproducibility |
| `draw_heatmap.py`          | Heatmap of word similarity across periods         |
| `visualize_pca_3d.py`      | 3D PCA of temporal word variants                  |
| `query_model.py`           | Query word similarities in a trained model        |
| `corpus_statistics.py`     | Print corpus statistics                           |


All scripts support `--help` for full options.

## Interpretation

- **High z-score (>3)**: Consistent across training runs
- **Low p-value (<0.05)**: Exceeds null expectation
- Words with both are strongest candidates for genuine semantic change

