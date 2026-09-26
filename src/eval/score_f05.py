"""
Local F_0.5 scorer, matching the challenge's exact formula:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed PER source1 entity, then macro-averaged across all entities.
A source1 entity with no true matches scores 1.0 if predicted empty,
0.0 if any match is predicted (false merge).

Usage:
    python score_f05.py \
        --predictions output/matching_results.tsv \
        --ground-truth dataset/val/val_ground_truth.tsv
"""

import argparse

import pandas as pd


def parse_id_list(s: str) -> set:
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: set, actual: set) -> float:
    if not actual and not predicted:
        return 1.0  # correct singleton
    if not predicted:
        precision = 1.0  # no denominator issue; recall determines score
        recall = 0.0 if actual else 1.0
    else:
        tp = len(predicted & actual)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(actual) if actual else 0.0
    return f_beta(precision, recall, beta=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--dump-errors", default=None,
                         help="optional path to write worst-scoring entities for error analysis")
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions, sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str).fillna("")

    pred_map = dict(zip(preds["source1_entity_id"], preds["matched_entity_ids"]))
    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))

    scores = []
    rows_for_dump = []
    for s1_id, actual_str in gt_map.items():
        predicted = parse_id_list(pred_map.get(s1_id, ""))
        actual = parse_id_list(actual_str)
        score = score_entity(predicted, actual)
        scores.append(score)
        rows_for_dump.append({
            "source1_entity_id": s1_id,
            "predicted": ",".join(sorted(predicted)),
            "actual": ",".join(sorted(actual)),
            "f0.5": round(score, 3),
        })

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    print(f"Entities scored: {len(scores)}")
    print(f"Macro F_0.5: {macro_f05:.4f}")

    n_perfect = sum(1 for s in scores if s == 1.0)
    n_zero = sum(1 for s in scores if s == 0.0)
    print(f"  Perfect (1.0): {n_perfect} ({100*n_perfect/len(scores):.1f}%)")
    print(f"  Zero (0.0):    {n_zero} ({100*n_zero/len(scores):.1f}%)")

    if args.dump_errors:
        dump_df = pd.DataFrame(rows_for_dump).sort_values("f0.5")
        dump_df.to_csv(args.dump_errors, sep="\t", index=False)
        print(f"Wrote per-entity breakdown (worst first) to {args.dump_errors}")


if __name__ == "__main__":
    main()
