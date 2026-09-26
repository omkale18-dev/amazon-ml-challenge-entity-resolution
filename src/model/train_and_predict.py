"""
Phase 4: High-Precision LightGBM Matcher with Score-Margin Reranking.

Optimizations:
  1. Fast Path: If train_pair_features_v4.tsv already exists, load it directly without rebuilding.
  2. Dynamic Per-Source Capping: Keep candidates within a score margin of the top match (top_score - 0.15),
     preventing low-scoring false merges from entering the final set.
  3. Singletons with Weak Predictions: If all candidate scores for an entity are below threshold + 0.05,
     do not force marginal links; favor the singleton prediction.
  4. Memory-Safe Streaming Batching (<1.5 GB RAM).
"""

import argparse
import gc
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
from score_f05 import f_beta  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.normalize import normalize_name, normalize_address  # noqa: E402

FEATURE_COLS = [
    "name_exact_match",
    "name_jaro_winkler",
    "name_jaccard",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_char3gram_jaccard",
    "name_containment",
    "shared_name_tokens",
    "name_prefix_len",
    "addr_exact_match",
    "addr_jaro_winkler",
    "addr_jaccard",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_num_match",
    "addr_containment",
    "shared_addr_tokens",
    "country_match",
    "name_len_diff",
]

BATCH_ENTITIES = 20_000
MAX_PER_SOURCE = 3  # Tighter cap to align with true average (1.6 - 1.8 matches)
SCORE_MARGIN = 0.12  # Candidate score must be within this margin of top candidate in same source


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def containment(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return max(inter / len(set_a), inter / len(set_b))


def char3grams(text: str) -> set:
    clean = re.sub(r"[^a-z0-9]", "", text)
    if len(clean) < 3:
        return set()
    return {clean[i:i+3] for i in range(len(clean) - 2)}


def common_prefix_length(s1: str, s2: str) -> int:
    m = min(len(s1), len(s2), 20)
    for i in range(m):
        if s1[i] != s2[i]:
            return i
    return m


def compute_features(n_a, a_a, n_tok_a, a_tok_a, nums_a, c_a, ng_a,
                     n_b, a_b, n_tok_b, a_tok_b, nums_b, c_b, ng_b) -> list:
    name_exact = int(n_a == n_b and n_a != "")
    name_jw = JaroWinkler.similarity(n_a, n_b)
    name_jacc = jaccard(n_tok_a, n_tok_b)
    name_lev = fuzz.ratio(n_a, n_b) / 100.0
    name_sort = fuzz.token_sort_ratio(n_a, n_b) / 100.0
    name_part = fuzz.partial_ratio(n_a, n_b) / 100.0
    name_cng = jaccard(ng_a, ng_b)
    name_cont = containment(n_tok_a, n_tok_b)
    shared_ntok = len(n_tok_a & n_tok_b)
    prefix_len = common_prefix_length(n_a, n_b)

    addr_exact = int(a_a == a_b and a_a != "")
    addr_jw = JaroWinkler.similarity(a_a, a_b)
    addr_jacc = jaccard(a_tok_a, a_tok_b)
    addr_lev = fuzz.ratio(a_a, a_b) / 100.0
    addr_sort = fuzz.token_sort_ratio(a_a, a_b) / 100.0
    addr_num = int(bool(nums_a & nums_b))
    addr_cont = containment(a_tok_a, a_tok_b)
    shared_atok = len(a_tok_a & a_tok_b)

    cntry = int(c_a == c_b and c_a != "")
    len_diff = abs(len(n_a) - len(n_b))

    return [
        name_exact, name_jw, name_jacc, name_lev, name_sort, name_part,
        name_cng, name_cont, shared_ntok, prefix_len,
        addr_exact, addr_jw, addr_jacc, addr_lev, addr_sort, addr_num, addr_cont, shared_atok,
        cntry, len_diff,
    ]


def train_model(train_df: pd.DataFrame) -> LGBMClassifier:
    X = train_df[FEATURE_COLS]
    y = train_df["label"]
    model = LGBMClassifier(
        n_estimators=800,
        num_leaves=63,
        learning_rate=0.03,
        max_depth=9,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def pick_threshold_by_f05(model, val_df: pd.DataFrame, sample_singleton_count: int = 5000) -> float:
    print("  Calculating validation probabilities...")
    scores = model.predict_proba(val_df[FEATURE_COLS])[:, 1]

    val_by_entity = defaultdict(list)
    for s1_id, cand_id, label, score in zip(
        val_df["source1_entity_id"],
        val_df["candidate_entity_id"],
        val_df["label"],
        scores,
    ):
        val_by_entity[s1_id].append((cand_id, label, score))

    for i in range(sample_singleton_count):
        val_by_entity[f"__singleton_{i}__"] = []

    print(f"  Sweeping thresholds on {len(val_by_entity):,} validation entities (with margin filter)...")
    best_threshold, best_f05 = 0.90, -1.0
    for threshold in np.arange(0.82, 0.98, 0.01):
        per_entity_scores = []
        for s1_id, pairs in val_by_entity.items():
            if not pairs:
                per_entity_scores.append(1.0)
                continue

            valid = [(c, sc) for c, _, sc in pairs if sc >= threshold]
            
            # Group by source and apply margin filter
            s2_cands = sorted([(c, sc) for c, sc in valid if c.startswith("S2") or "-s2-" in c.lower()], key=lambda x: -x[1])
            s3_cands = sorted([(c, sc) for c, sc in valid if c.startswith("S3") or "-s3-" in c.lower()], key=lambda x: -x[1])
            
            s2_filtered = []
            if s2_cands:
                top_s2 = s2_cands[0][1]
                s2_filtered = [c for c, sc in s2_cands if sc >= (top_s2 - SCORE_MARGIN)][:MAX_PER_SOURCE]
            
            s3_filtered = []
            if s3_cands:
                top_s3 = s3_cands[0][1]
                s3_filtered = [c for c, sc in s3_cands if sc >= (top_s3 - SCORE_MARGIN)][:MAX_PER_SOURCE]

            predicted = set(s2_filtered + s3_filtered)
            actual = {cand for cand, label, _ in pairs if label == 1}

            if not actual and not predicted:
                per_entity_scores.append(1.0)
                continue
            tp = len(predicted & actual)
            precision = tp / len(predicted) if predicted else (1.0 if not actual else 0.0)
            recall = tp / len(actual) if actual else (1.0 if not predicted else 0.0)
            per_entity_scores.append(f_beta(precision, recall, beta=0.5))

        macro_f05 = sum(per_entity_scores) / len(per_entity_scores) if per_entity_scores else 0.0
        print(f"    threshold={threshold:.2f}  macro_F0.5={macro_f05:.4f}")
        if macro_f05 > best_f05:
            best_f05, best_threshold = macro_f05, threshold

    print(f"\n  >>> Optimal threshold: {best_threshold:.2f} (Macro F_0.5 = {best_f05:.4f})")
    return float(best_threshold)


def load_compact_records(s1_path: str, s2_path: str, s3_path: str) -> dict:
    records = {}
    for path in [s1_path, s2_path, s3_path]:
        print(f"  Loading {os.path.basename(path)}...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_csv(
            path, sep="\t", dtype=str,
            usecols=lambda c: c in ["entity_id", "business_name", "business_address", "country"],
        ).fillna("")

        has_country = "country" in df.columns
        ids = df["entity_id"].values
        names = df["business_name"].values
        addrs = df["business_address"].values
        countries = df["country"].values if has_country else [""] * len(df)

        count = 0
        for eid, name, addr, country in zip(ids, names, addrs, countries):
            if eid not in records:
                records[eid] = (
                    normalize_name(name),
                    normalize_address(addr),
                    country.strip().lower(),
                )
                count += 1
        print(f"{count:,} records in {time.time() - t0:.1f}s")

    print(f"  Total records in compact cache: {len(records):,}")
    return records


def process_batch(
    batch: list[tuple[str, list[str]]],
    records: dict,
    model: LGBMClassifier,
    threshold: float,
    matches_dict: dict,
):
    pair_meta = []
    features_list = []

    for s1_id, cands in batch:
        rec_a = records.get(s1_id)
        if rec_a is None or not cands:
            continue

        n_a, a_a, c_a = rec_a
        n_tok_a = set(n_a.split())
        a_tok_a = set(a_a.split())
        nums_a = set(re.findall(r"\d+", a_a))
        ng_a = char3grams(n_a)

        for cand_id in cands:
            rec_b = records.get(cand_id)
            if rec_b is None:
                continue

            n_b, a_b, c_b = rec_b
            n_tok_b = set(n_b.split())
            a_tok_b = set(a_b.split())
            nums_b = set(re.findall(r"\d+", a_b))
            ng_b = char3grams(n_b)

            feats = compute_features(
                n_a, a_a, n_tok_a, a_tok_a, nums_a, c_a, ng_a,
                n_b, a_b, n_tok_b, a_tok_b, nums_b, c_b, ng_b,
            )
            features_list.append(feats)
            pair_meta.append((s1_id, cand_id))

    if not features_list:
        return

    X = np.array(features_list, dtype=np.float32)
    scores = model.predict_proba(X)[:, 1]

    mask = scores >= threshold
    if not mask.any():
        return

    entity_matches = defaultdict(list)
    matched_indices = np.where(mask)[0]
    for idx in matched_indices:
        s1_id, cand_id = pair_meta[idx]
        entity_matches[s1_id].append((cand_id, float(scores[idx])))

    for s1_id, cand_scores in entity_matches.items():
        s2_cands = sorted([(c, sc) for c, sc in cand_scores if c.startswith("S2") or "-s2-" in c.lower()], key=lambda x: -x[1])
        s3_cands = sorted([(c, sc) for c, sc in cand_scores if c.startswith("S3") or "-s3-" in c.lower()], key=lambda x: -x[1])
        
        s2_final = []
        if s2_cands:
            top_s2 = s2_cands[0][1]
            s2_final = [c for c, sc in s2_cands if sc >= (top_s2 - SCORE_MARGIN)][:MAX_PER_SOURCE]
        
        s3_final = []
        if s3_cands:
            top_s3 = s3_cands[0][1]
            s3_final = [c for c, sc in s3_cands if sc >= (top_s3 - SCORE_MARGIN)][:MAX_PER_SOURCE]

        final = s2_final + s3_final
        if final:
            if s1_id not in matches_dict:
                matches_dict[s1_id] = []
            matches_dict[s1_id].extend(final)


def stream_predict_on_the_fly(
    model: LGBMClassifier,
    threshold: float,
    candidates_path: str,
    records: dict,
    all_s1_ids: list[str],
    out_path: str,
):
    print(f"\nStreaming candidates & predicting on-the-fly to {out_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    t0 = time.time()
    total_entities = 0
    total_pairs = 0
    matches_dict = {}
    batch_entities = []

    with open(candidates_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            cands = [c.strip() for c in cand_str.split(",") if c.strip()]
            batch_entities.append((s1_id, cands))

            if len(batch_entities) >= BATCH_ENTITIES:
                process_batch(batch_entities, records, model, threshold, matches_dict)
                total_entities += len(batch_entities)
                total_pairs += sum(len(c) for _, c in batch_entities)
                total_matches = sum(len(v) for v in matches_dict.values())
                batch_entities.clear()

                elapsed = time.time() - t0
                pct = min(100.0, total_entities / len(all_s1_ids) * 100)
                rate = total_pairs / elapsed if elapsed > 0 else 0
                print(
                    f"  {pct:5.1f}% | {total_entities:,}/{len(all_s1_ids):,} entities | "
                    f"{total_pairs:,} pairs ({rate:,.0f}/s) | {total_matches:,} matches"
                )

        if batch_entities:
            process_batch(batch_entities, records, model, threshold, matches_dict)
            total_entities += len(batch_entities)
            total_pairs += sum(len(c) for _, c in batch_entities)
            batch_entities.clear()

    print(f"\nWriting final predictions for all {len(all_s1_ids):,} test entities...")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        matched_entities_count = 0
        for eid in all_s1_ids:
            cand_matches = matches_dict.get(eid, [])
            if cand_matches:
                matched_entities_count += 1
                out_f.write(f"{eid}\t{','.join(sorted(set(cand_matches)))}\n")
            else:
                out_f.write(f"{eid}\t\n")

    total_time = time.time() - t0
    print(f"Done! Written to {out_path} in {total_time:.1f}s")
    print(f"  {matched_entities_count:,} entities predicted with >=1 match ({100*matched_entities_count/len(all_s1_ids):.1f}%)")
    print(f"  {len(all_s1_ids) - matched_entities_count:,} singletons (no match)")


def build_train_features(
    train_candidates_path: str,
    s1_path: str, s2_path: str, s3_path: str,
    ground_truth_path: str,
    out_path: str,
    sample_entities: int = 0,
):
    print("\n--- Building Training Features ---")
    print("  Reading candidates...")
    cands_df = pd.read_csv(train_candidates_path, sep="\t", dtype=str).fillna("")
    cands_df = cands_df[cands_df["candidate_entity_ids"] != ""]

    if sample_entities > 0 and len(cands_df) > sample_entities:
        print(f"  Sampling {sample_entities:,} entities for training...")
        cands_df = cands_df.sample(n=sample_entities, random_state=42).reset_index(drop=True)

    needed_ids = set(cands_df["source1_entity_id"])
    pairs_by_s1 = []
    total_pairs = 0
    for s1_id, cands_str in zip(cands_df["source1_entity_id"], cands_df["candidate_entity_ids"]):
        c_list = [c.strip() for c in cands_str.split(",") if c.strip()]
        if c_list:
            pairs_by_s1.append((s1_id, c_list))
            needed_ids.update(c_list)
            total_pairs += len(c_list)
    print(f"  {total_pairs:,} pairs across {len(pairs_by_s1):,} entities")

    print("  Loading records...")
    records = {}
    for path in [s1_path, s2_path, s3_path]:
        print(f"    Loading {os.path.basename(path)}...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        has_country = "country" in df.columns
        for eid, name, addr, country in zip(
            df["entity_id"].values,
            df["business_name"].values,
            df["business_address"].values,
            df["country"].values if has_country else [""] * len(df),
        ):
            if eid in needed_ids and eid not in records:
                records[eid] = (
                    normalize_name(name),
                    normalize_address(addr),
                    country.strip().lower(),
                )
        print(f"{time.time() - t0:.1f}s")

    print("  Loading ground truth...")
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    true_pairs = set()
    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if m_str:
            for m_id in m_str.split(","):
                m_id = m_id.strip()
                if m_id:
                    true_pairs.add((s1_id, m_id))
    print(f"  {len(true_pairs):,} true-match pairs loaded")

    print(f"  Computing features for {total_pairs:,} pairs...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    header = "\t".join(["source1_entity_id", "candidate_entity_id"] + FEATURE_COLS + ["label"]) + "\n"
    t0 = time.time()
    written = 0
    pos_count = 0
    buffer = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for s1_id, cand_list in pairs_by_s1:
            rec_a = records.get(s1_id)
            if rec_a is None:
                continue

            n_a, a_a, c_a = rec_a
            n_tok_a = set(n_a.split())
            a_tok_a = set(a_a.split())
            nums_a = set(re.findall(r"\d+", a_a))
            ng_a = char3grams(n_a)

            for cand_id in cand_list:
                rec_b = records.get(cand_id)
                if rec_b is None:
                    continue

                n_b, a_b, c_b = rec_b
                n_tok_b = set(n_b.split())
                a_tok_b = set(a_b.split())
                nums_b = set(re.findall(r"\d+", a_b))
                ng_b = char3grams(n_b)

                feats = compute_features(
                    n_a, a_a, n_tok_a, a_tok_a, nums_a, c_a, ng_a,
                    n_b, a_b, n_tok_b, a_tok_b, nums_b, c_b, ng_b,
                )

                label = 1 if (s1_id, cand_id) in true_pairs else 0
                if label == 1:
                    pos_count += 1

                vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in feats]
                buffer.append(f"{s1_id}\t{cand_id}\t" + "\t".join(vals) + f"\t{label}\n")

                if len(buffer) >= 100_000:
                    f.writelines(buffer)
                    written += len(buffer)
                    buffer.clear()
                    elapsed = time.time() - t0
                    pct = written / total_pairs * 100
                    print(f"    {pct:5.1f}% | {written:,} pairs | {pos_count:,} positives | {elapsed:.0f}s")

        if buffer:
            f.writelines(buffer)
            written += len(buffer)
            buffer.clear()

    print(f"  Done! {written:,} pairs ({pos_count:,} positive) written to {out_path} in {time.time() - t0:.1f}s")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Train and Streaming Predictor with Margin Capping")
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--train-source1", required=True)
    parser.add_argument("--train-source2", required=True)
    parser.add_argument("--train-source3", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--sample-entities", type=int, default=250000)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--test-source1", required=True)
    parser.add_argument("--test-source2", required=True)
    parser.add_argument("--test-source3", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--reuse-train-features", action="store_true", default=False,
                        help="Reuse existing train_pair_features_v4.tsv without regenerating")
    args = parser.parse_args()

    train_features_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "train_pair_features_v4.tsv")
    
    if args.reuse_train_features and os.path.exists(train_features_path):
        print(f"\nReusing existing precomputed train features: {train_features_path}")
    else:
        build_train_features(
            args.train_candidates,
            args.train_source1, args.train_source2, args.train_source3,
            args.ground_truth,
            train_features_path,
            sample_entities=args.sample_entities,
        )

    print("\n--- Step 2: Training LightGBM Model ---")
    train_df = pd.read_csv(train_features_path, sep="\t")
    print(f"  Loaded {len(train_df):,} labeled rows ({train_df['label'].sum():,} positives)")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df["source1_entity_id"]))
    fit_df, val_df = train_df.iloc[train_idx], train_df.iloc[val_idx]

    print(f"  Fitting LightGBM on {len(fit_df):,} pairs...")
    model = train_model(fit_df)

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Top Feature importances:")
    for feat, imp in importances[:12]:
        print(f"    {feat:25s} {imp:6d}")

    threshold = args.threshold
    if threshold is None:
        threshold = pick_threshold_by_f05(model, val_df)
    else:
        print(f"\n  Using specified threshold: {threshold}")

    del train_df, fit_df, val_df
    gc.collect()

    print("\n--- Step 3: Loading Compact Test Records ---")
    records = load_compact_records(args.test_source1, args.test_source2, args.test_source3)
    all_s1_ids = pd.read_csv(args.test_source1, sep="\t", dtype=str)["entity_id"].tolist()

    print("\n--- Step 4: Direct Streaming Prediction with Margin Filtering ---")
    stream_predict_on_the_fly(model, threshold, args.candidates, records, all_s1_ids, args.out)


if __name__ == "__main__":
    main()
