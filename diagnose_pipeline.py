"""
Diagnostic script — run this against your TRAIN split to find exactly what's
costing you points between 0.848 and the 0.95+ top teams are getting.

It answers three questions in order of importance:

1. BLOCKING RECALL CEILING: what fraction of true matches are even present
   in candidate_pairs.tsv? If this is below ~98%, no amount of model tuning
   can fix it -- you need better blocking, full stop.

2. SINGLETON PRECISION: since F_0.5 gives a full 1.0 for correct singletons
   and 0.0 for any false merge on one, this is usually the single biggest
   lever. What fraction of true singletons are you incorrectly matching?

3. WORST-SCORING ENTITIES: dumps the specific entities dragging your score
   down the most, with their predicted vs actual matches, so you can look
   at the real records and see the pattern (chain-store false merges?
   transliteration misses? address-only false positives?).

Usage:
    python diagnose_pipeline.py \
        --candidates output/train_candidate_pairs.tsv \
        --predictions output/train_matching_results.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --dump-worst output/worst_entities.tsv \
        --n-worst 100

Notes:
  - --predictions should come from running your pipeline on a TRAIN split
    (e.g. a held-out validation slice) where you have ground truth to
    compare against. Running this against your real test predictions is
    not possible since you don't have test ground truth.
  - If you don't have --predictions yet, this script still runs steps 1
    (candidate recall) using just --candidates and --ground-truth.
"""

import argparse
import sys

import pandas as pd


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def parse_ids(s) -> set:
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def load_ground_truth(path: str) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))


def check_blocking_recall(candidates_path: str, gt_map: dict):
    print("=" * 70)
    print("STEP 1: BLOCKING RECALL CEILING")
    print("=" * 70)

    cands = pd.read_csv(candidates_path, sep="\t", dtype=str).fillna("")
    cand_map = dict(zip(cands["source1_entity_id"], cands["candidate_entity_ids"]))

    total_true_matches = 0
    matches_covered = 0
    entities_with_missed_match = []

    for s1_id, actual_str in gt_map.items():
        actual = parse_ids(actual_str)
        if not actual:
            continue  # singleton, nothing to check for blocking recall
        candidate_set = parse_ids(cand_map.get(s1_id, ""))
        covered = actual & candidate_set
        missed = actual - candidate_set

        total_true_matches += len(actual)
        matches_covered += len(covered)
        if missed:
            entities_with_missed_match.append((s1_id, missed, len(candidate_set)))

    recall_ceiling = matches_covered / total_true_matches if total_true_matches else 1.0
    print(f"Entities with >=1 true match: {sum(1 for v in gt_map.values() if parse_ids(v))}")
    print(f"Total true match pairs: {total_true_matches}")
    print(f"True match pairs covered by candidates: {matches_covered}")
    print(f"*** BLOCKING RECALL CEILING: {recall_ceiling:.4f} ***")
    print(f"Entities with at least one missed true match: {len(entities_with_missed_match)}")

    if recall_ceiling < 0.98:
        print()
        print("[!] RECALL CEILING IS BELOW 0.98 -- this alone caps your maximum")
        print("    possible F_0.5. No model/threshold tuning can recover matches")
        print("    that never made it into candidate_pairs.tsv. Fix blocking first.")
    else:
        print()
        print("[+] Recall ceiling looks solid. Your score gap is likely in the")
        print("   MODEL/THRESHOLD stage (precision, singleton handling), not blocking.")

    if entities_with_missed_match:
        print(f"\nSample of entities with missed matches (up to 15):")
        for s1_id, missed, n_cands in entities_with_missed_match[:15]:
            print(f"  {s1_id}: missed {missed} (had {n_cands} candidates)")

    return recall_ceiling, entities_with_missed_match


def check_predictions(predictions_path: str, gt_map: dict, dump_worst_path: str, n_worst: int):
    print()
    print("=" * 70)
    print("STEP 2 & 3: PREDICTION QUALITY (singleton precision + worst entities)")
    print("=" * 70)

    preds = pd.read_csv(predictions_path, sep="\t", dtype=str).fillna("")
    pred_map = dict(zip(preds["source1_entity_id"], preds["matched_entity_ids"]))

    rows = []
    singleton_correct = 0
    singleton_total = 0
    singleton_false_merge = 0
    matched_entity_scores = []

    for s1_id, actual_str in gt_map.items():
        actual = parse_ids(actual_str)
        predicted = parse_ids(pred_map.get(s1_id, ""))

        if not actual:
            singleton_total += 1
            if not predicted:
                singleton_correct += 1
                score = 1.0
            else:
                singleton_false_merge += 1
                score = 0.0
        else:
            tp = len(predicted & actual)
            precision = tp / len(predicted) if predicted else (1.0 if not actual else 0.0)
            recall = tp / len(actual) if actual else 0.0
            score = f_beta(precision, recall, beta=0.5)
            matched_entity_scores.append(score)

        rows.append({
            "source1_entity_id": s1_id,
            "predicted": ",".join(sorted(predicted)),
            "actual": ",".join(sorted(actual)),
            "is_singleton": len(actual) == 0,
            "f0.5": round(score, 3),
        })

    all_scores = [r["f0.5"] for r in rows]
    macro_f05 = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"Overall macro F_0.5: {macro_f05:.4f}")
    print()
    print(f"--- Singleton breakdown ({singleton_total} true singletons) ---")
    print(f"  Correctly predicted empty: {singleton_correct} "
          f"({100*singleton_correct/singleton_total:.1f}%)" if singleton_total else "  (none)")
    print(f"  FALSE MERGES on singletons: {singleton_false_merge} "
          f"({100*singleton_false_merge/singleton_total:.1f}%)" if singleton_total else "")
    if singleton_false_merge > 0:
        print(f"  ⚠️  Each false merge on a singleton costs a full 1.0 -> 0.0 point.")
        print(f"      {singleton_false_merge} false merges = {singleton_false_merge} points lost here alone.")

    print()
    print(f"--- Non-singleton (real match) entities ({len(matched_entity_scores)}) ---")
    avg_matched_score = sum(matched_entity_scores) / len(matched_entity_scores) if matched_entity_scores else 0
    print(f"  Average F_0.5 on entities with real matches: {avg_matched_score:.4f}")

    dump_df = pd.DataFrame(rows).sort_values("f0.5")
    dump_df.head(n_worst).to_csv(dump_worst_path, sep="\t", index=False)
    print(f"\nWrote {n_worst} worst-scoring entities to {dump_worst_path}")
    print("Open this file and look at 'predicted' vs 'actual' for patterns:")
    print("  - predicted has EXTRA ids not in actual -> false merges (precision problem)")
    print("  - predicted is MISSING ids from actual -> missed matches (recall/blocking problem)")
    print("  - Cross-reference these entity_ids against source1/2/3 records to see WHY")
    print("    (shared address but different business? transliteration? chain-store name?)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", default=None,
                         help="optional: your matching_results.tsv run on this same train/val split")
    parser.add_argument("--dump-worst", default="worst_entities.tsv")
    parser.add_argument("--n-worst", type=int, default=100)
    args = parser.parse_args()

    gt_map = load_ground_truth(args.ground_truth)

    check_blocking_recall(args.candidates, gt_map)

    if args.predictions:
        check_predictions(args.predictions, gt_map, args.dump_worst, args.n_worst)
    else:
        print()
        print("(Skipping prediction-quality checks -- pass --predictions to also see")
        print(" singleton precision and worst-scoring entities.)")


if __name__ == "__main__":
    main()
