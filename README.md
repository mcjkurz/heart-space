# Interiority Semantic Change Analysis

Semantic change in Chinese interiority words (心里, 心中, 内心, and the rest of `data/dictionaries/interiority_words.txt`) across five periods, using temporal referencing Word2Vec (TempRef).

Run commands from the project root. The scripts generate `models/`, `results/`, and `images/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Periods

| Period       | Label          | Approximate dates |
| ------------ | -------------- | ----------------- |
| Ming-Qing    | `mingqing`     | 1368–1860         |
| Late Qing    | `late_qing`    | 1860–1911         |
| Republican   | `republican`   | 1911–1949         |
| Socialist    | `socialist`    | 1949–1976         |
| Contemporary | `contemporary` | 1976–present      |

The label is the folder name under `data/texts/`. Each script has the same list, in this order, as `PERIODS` near the top of the file. Later steps look for `data/texts_normalized/{label}/` and `data/segmented/sentences_{label}.txt`.

## Pipeline

### 1. Normalize texts

```bash
python scripts/normalize_texts.py
```

Converts traditional characters to simplified, removes noise, and normalizes punctuation. Output: `data/texts_normalized/`.

### 2. Segment texts

```bash
python scripts/segment_texts.py
```

Segments with dictionary-enhanced jieba. Output: `data/segmented/sentences_{period}.txt`.

### 3. Train models

```bash
python scripts/train_tempref.py \
  --trials 100 \
  --processes 4 \
  --output-dir models/real/

python scripts/train_tempref.py \
  --trials 100 \
  --processes 4 \
  --permute-periods \
  --output-dir models/null/
```

The first command trains the real models. The second shuffles period labels and trains the null models used for p-values. Files are named `model_e{epochs}_s{seed}.npy` (`_null.npy` for the shuffled run). Omit `--trials` to train a single model.

### 4. Measure semantic change

```bash
python scripts/semantic_change.py \
  --model-dir models/real/ \
  --null-dir models/null/ \
  --output-dir results/multiseed/
```

Writes one CSV per adjacent transition, for example `semantic_changes_late_qing_to_republican.csv`. Each row is a neighboring word. `mean_change` is the average score across the 100 models, `z_score` is that mean divided by the standard deviation, and `p_value` compares the mean with the null models.

Every part of speech is kept. Pass `--postag "n.*"` to keep only nouns. Words below the frequency cutoffs are still dropped (`--min-count 5` in both periods, `--min-cooc 5` with the interiority words in the later period).

A high z-score (above 3) means the change is steady across training runs. A low p-value (below 0.05) means it exceeds the null models. Words with both are the strongest candidates.

For one model, and for the histogram and single-model heatmap that go with it:

```bash
python scripts/semantic_change.py --model models/real/model_e3_s42.npy
```

## Figures

Plots go to `images/`.

```bash
# Period-similarity heatmap, averaged over models/real/
python scripts/draw_heatmap.py --model-dir models/real/

# 3D PCA trajectory for one model
python scripts/visualize_pca_3d.py --model models/real/model_e3_s42.npy

# One trajectory per model (null models skipped)
python scripts/visualize_pca_3d.py \
  --model-dir models/real/ \
  --output-dir images/pca/

# Where interiority words fall inside 蹉跎岁月
python experiments/interiority_distribution.py
```

`images/characters_heart_bar.png` comes from `experiments/Wasted_Years_DH_Analysis.ipynb`. Run that notebook from `experiments/`.

## Stability

**Ensemble size.** Split `models/real/` into two groups of size k and compare their top-100 lists. Defaults are k = 1, 3, 5, 10, and 20, with 200 random splits each.

```bash
python scripts/ensemble_stability.py \
  --model-dir models/real/ \
  --output-dir results/ensemble_stability/
```

**Epoch count.** Train several seeds at each epoch setting, then compare agreement within a setting and between settings.

```bash
python scripts/train_tempref.py \
  --trials 10 --epochs 1 --seed-start 1 \
  --output-dir models/epochs/

python scripts/train_tempref.py \
  --trials 10 --epochs 3 --seed-start 1 \
  --output-dir models/epochs/

python scripts/train_tempref.py \
  --trials 10 --epochs 5 --seed-start 1 \
  --output-dir models/epochs/

python scripts/train_tempref.py \
  --trials 10 --epochs 7 --seed-start 1 \
  --output-dir models/epochs/

python scripts/epoch_validation.py \
  --model-dir models/epochs/ \
  --output-dir results/epoch_validation/
```

**Stable words.** Trains its own models. Keeps words that occur at least `--min-freq` times in every period (default 100), then ranks them by how similar their vector stays from one period to the next. Higher `mean_sim_mean` means more stable. The trial models are not saved. Scores are kept in `results/word_stability_scores_checkpoint.npz` until the run finishes, then that file is deleted. `--resume` continues from it. One final model, trained with the last seed, is saved.

```bash
python scripts/find_stable_words.py \
  --trials 100 \
  --processes 4 \
  --min-freq 100
```

Writes `results/word_stability_scores.csv` and `models/tempref_stable_words.npy`. Pass `--output` to save that model somewhere else.

## Other scripts

Separate Word2Vec model for each period, then nearest neighbors of the target:

```bash
python scripts/train_period_models.py --trials 10 --output-dir models/period_models/
python scripts/analyze_period_models.py --model-dir models/period_models/
```

Look up one word in one trained model. Pass that file with `--model`. The comparison is to the `interiority` target in each period.

```bash
python scripts/query_model.py \
  --model models/real/model_e3_s42.npy \
  --query 大人
```

If `--model` is omitted, the script loads `models/tempref_interiority_w2v.npy`.

Print corpus statistics:

```bash
python scripts/corpus_statistics.py
```

The same comparison with PPMI, then the words that rank in the top 100 of both methods:

```bash
python experiments/ppmi/ppmi_semantic_change.py
python experiments/ppmi/compare_methods.py results/ppmi results/multiseed --topn 100
```

Replace pronouns in 蹉跎岁月 with character names:

```bash
python experiments/suiyue.py
```

Every script accepts `--help`.
