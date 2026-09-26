# Business Entity Resolution — Pipeline

Baseline end-to-end pipeline: blocking -> feature engineering -> gradient-boosted
matching model -> threshold tuned for F_0.5.

## Setup

```bash
pip install -r requirements.txt
```

Expects the data in this layout (as provided by the challenge):

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

## Run end-to-end

All commands are run from `src/`, and paths below are relative to the
repo root. Run in this exact order (each stage writes files the next
stage reads):

```bash
cd src

# 1. Blocking — build candidate pairs for TRAIN (needed to train the model)
python blocking/generate_candidates.py \
    --source1 ../dataset/train/train_source1.tsv \
    --source2 ../dataset/train/train_source2.tsv \
    --source3 ../dataset/train/train_source3.tsv \
    --out ../output/train_candidate_pairs.tsv

# 2. Blocking — build candidate pairs for TEST (this is your submitted candidate_pairs.tsv)
python blocking/generate_candidates.py \
    --source1 ../dataset/test/test_source1.tsv \
    --source2 ../dataset/test/test_source2.tsv \
    --source3 ../dataset/test/test_source3.tsv \
    --out ../output/candidate_pairs.tsv

# 3. Features for TRAIN pairs
python features/build_features.py \
    --source1 ../dataset/train/train_source1.tsv \
    --source2 ../dataset/train/train_source2.tsv \
    --source3 ../dataset/train/train_source3.tsv \
    --candidates ../output/train_candidate_pairs.tsv \
    --out ../output/train_pair_features.tsv

# 4. Features for TEST pairs
python features/build_features.py \
    --source1 ../dataset/test/test_source1.tsv \
    --source2 ../dataset/test/test_source2.tsv \
    --source3 ../dataset/test/test_source3.tsv \
    --candidates ../output/candidate_pairs.tsv \
    --out ../output/test_pair_features.tsv

# 5. Label the TRAIN pairs using ground truth
python model/label_pairs.py \
    --features ../output/train_pair_features.tsv \
    --ground-truth ../dataset/train/train_ground_truth.tsv \
    --out ../output/train_pair_features_labeled.tsv

# 6. Train model + predict on TEST -> this produces matching_results.tsv
python model/train_and_predict.py \
    --train-features ../output/train_pair_features_labeled.tsv \
    --test-features ../output/test_pair_features.tsv \
    --test-source1 ../dataset/test/test_source1.tsv \
    --out ../output/matching_results.tsv
```

At this point `output/matching_results.tsv` and `output/candidate_pairs.tsv`
are ready to submit.

## Validate before submitting

Run the challenge's own validator (place `validate_submission.py` from
the challenge's `utils/` folder wherever you keep it, then):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Scoring locally against a held-out split

To measure your own F_0.5 before submitting, hold out part of the
training data as a validation set (with its own ground truth), run the
same 6 steps above pointing at that split instead of `test/`, then:

```bash
python eval/score_f05.py \
    --predictions ../output/val_matching_results.tsv \
    --ground-truth ../dataset/val/val_ground_truth.tsv \
    --dump-errors ../output/val_error_breakdown.tsv
```

`val_error_breakdown.tsv` lists every entity sorted worst-scoring first —
use it for error analysis to see which noise patterns are hurting score.

## Pipeline overview

1. **Blocking** (`blocking/generate_candidates.py`): token-overlap +
   TF-IDF character n-gram cosine similarity over normalized business
   names. Deliberately recall-favoring; the model stage handles precision.
2. **Features** (`features/build_features.py`): name/address similarity
   features (Jaccard, Levenshtein ratio, token-sort ratio, partial ratio)
   plus a country-match flag.
3. **Model** (`model/train_and_predict.py`): LightGBM classifier trained
   on labeled candidate pairs from the training set, with `class_weight="balanced"`
   since true matches are a minority of candidates. Threshold is swept
   on a held-out slice to maximize macro F_0.5 directly (not accuracy/F1).

## Known limitations (baseline — iterate on these)

- Country is used only as a soft feature (not a hard filter), which is
  intentional so unseen countries (e.g., France in the test set) still work.
- Blocking uses only `business_name` for the TF-IDF pass; adding an
  address-based blocking signal should improve recall further.
- No cross-account / distributed-training complexity — trains in a single
  process, fine for the ~1GB data volume on `ml.m5.xlarge`-class instances.
