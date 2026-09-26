"""
Attach ground-truth labels (1 = true match, 0 = not a match) to a
training pair-features file, based on train_ground_truth.tsv.

v2: vectorized ground-truth loading (no iterrows).

Usage:
    python label_pairs.py --features ../output/train_pair_features.tsv --ground-truth ../dataset/train/train_ground_truth.tsv --out ../output/train_pair_features_labeled.tsv
"""

import argparse

import pandas as pd


def load_ground_truth_as_set(path: str) -> set:
    """Returns a set of (source1_entity_id, matched_entity_id) true-match pairs."""
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    gt = gt[gt["matched_entity_ids"] != ""]

    # Vectorized explode instead of iterrows
    gt["matched_entity_id"] = gt["matched_entity_ids"].str.split(",")
    exploded = gt[["source1_entity_id", "matched_entity_id"]].explode("matched_entity_id")
    exploded["matched_entity_id"] = exploded["matched_entity_id"].str.strip()
    exploded = exploded[exploded["matched_entity_id"] != ""]

    pairs = set(zip(exploded["source1_entity_id"], exploded["matched_entity_id"]))
    print(f"  {len(pairs):,} true-match pairs loaded from ground truth")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading ground truth...")
    true_pairs = load_ground_truth_as_set(args.ground_truth)

    print("Loading features and labeling...")
    features = pd.read_csv(args.features, sep="\t")

    # Vectorized labeling using a set lookup via apply (fast with sets)
    features["label"] = [
        int((s1, cand) in true_pairs)
        for s1, cand in zip(features["source1_entity_id"].astype(str),
                            features["candidate_entity_id"].astype(str))
    ]

    features.to_csv(args.out, sep="\t", index=False)
    n_pos = features["label"].sum()
    print(f"Labeled {len(features):,} pairs: {n_pos:,} positive, "
          f"{len(features) - n_pos:,} negative.")


if __name__ == "__main__":
    main()
