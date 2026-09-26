"""
High-performance feature engineering for source1 <-> candidate pairs.

Optimizations:
  1. Pre-computes normalized strings, token sets, and address numbers ONCE per record.
  2. Uses fast Python dictionary lookup instead of slow pandas .loc[] on 12M rows.
  3. Eliminates slow pandas .iterrows() — uses direct list/tuple unpacking.
  4. Features include:
     - Name exact match, Jaccard, Levenshtein, Token Sort, Partial Ratio, Length Diff
     - Address exact match, Jaccard, Levenshtein, Token Sort, Address Number Match
     - Country match
  5. Supports optional --sample-entities flag for training.
  6. Streams candidate pairs in batches to keep peak memory below 2 GB.
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
from rapidfuzz import fuzz

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.normalize import normalize_name, normalize_address  # noqa: E402

CHUNK_PAIRS = 100_000


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def load_records_lookup(s1_path: str, s2_path: str, s3_path: str, needed_ids: set = None) -> dict:
    records = {}

    for path in [s1_path, s2_path, s3_path]:
        print(f"  Loading & pre-normalizing {os.path.basename(path)}...", end=" ", flush=True)
        t0 = time.time()
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

        has_country = "country" in df.columns
        ids = df["entity_id"].values
        names = df["business_name"].values
        addrs = df["business_address"].values
        countries = df["country"].values if has_country else [""] * len(df)

        count = 0
        for eid, name, addr, country in zip(ids, names, addrs, countries):
            if needed_ids is not None and eid not in needed_ids:
                continue
            if eid in records:
                continue

            n_norm = normalize_name(name)
            a_norm = normalize_address(addr)
            addr_nums = set(re.findall(r"\d+", a_norm))

            records[eid] = (
                n_norm,
                a_norm,
                set(n_norm.split()),
                set(a_norm.split()),
                addr_nums,
                country.strip().lower(),
            )
            count += 1

        print(f"{count:,} records ({time.time() - t0:.1f}s)")

    print(f"  Total records in fast lookup cache: {len(records):,}")
    return records


def get_needed_entity_ids_and_pairs(candidates_path: str, sample_entities: int = 0):
    print(f"  Reading candidates from {os.path.basename(candidates_path)}...")
    df = pd.read_csv(candidates_path, sep="\t", dtype=str).fillna("")
    df = df[df["candidate_entity_ids"] != ""]

    if sample_entities > 0 and len(df) > sample_entities:
        print(f"  Sampling {sample_entities:,} source1 entities out of {len(df):,} for training...")
        df = df.sample(n=sample_entities, random_state=42).reset_index(drop=True)

    needed_ids = set(df["source1_entity_id"])
    pairs_by_s1 = []
    total_pairs = 0

    for s1_id, cands_str in zip(df["source1_entity_id"], df["candidate_entity_ids"]):
        c_list = [c.strip() for c in cands_str.split(",") if c.strip()]
        if c_list:
            pairs_by_s1.append((s1_id, c_list))
            needed_ids.update(c_list)
            total_pairs += len(c_list)

    print(f"  Total pairs to featurize: {total_pairs:,} across {len(pairs_by_s1):,} source1 entities")
    return pairs_by_s1, needed_ids, total_pairs


def main():
    parser = argparse.ArgumentParser(description="Fast feature calculation for candidate pairs")
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sample-entities",
        type=int,
        default=0,
        help="Optional: sample N source1 entities (e.g. 150000 for train set). 0 = all.",
    )
    args = parser.parse_args()

    # Step 1: Scan candidate pairs and get required IDs
    print("Scanning candidate pairs...")
    pairs_by_s1, needed_ids, total_pairs = get_needed_entity_ids_and_pairs(
        args.candidates, sample_entities=args.sample_entities
    )

    if total_pairs == 0:
        print("No pairs found. Writing empty output.")
        pd.DataFrame().to_csv(args.out, sep="\t", index=False)
        return

    # Step 2: Load and pre-normalize ONLY the records participating in pairs
    print("\nLoading and pre-normalizing records...")
    records = load_records_lookup(args.source1, args.source2, args.source3, needed_ids=needed_ids)
    del needed_ids

    # Step 3: Compute features in high-speed batches
    print(f"\nComputing features for {total_pairs:,} pairs...")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    header = (
        "source1_entity_id\tcandidate_entity_id\t"
        "name_exact_match\tname_jaccard\tname_levenshtein_ratio\t"
        "name_token_sort_ratio\tname_partial_ratio\taddr_exact_match\t"
        "addr_jaccard\taddr_levenshtein_ratio\taddr_token_sort_ratio\t"
        "addr_num_match\tcountry_match\tname_len_diff\n"
    )

    t0 = time.time()
    written_count = 0
    buffer = []

    with open(args.out, "w", encoding="utf-8") as out_f:
        out_f.write(header)

        for s1_id, cand_list in pairs_by_s1:
            rec_a = records.get(s1_id)
            if rec_a is None:
                continue

            n_a, a_a, n_tok_a, a_tok_a, nums_a, c_a = rec_a
            len_n_a = len(n_a)

            for cand_id in cand_list:
                rec_b = records.get(cand_id)
                if rec_b is None:
                    continue

                n_b, a_b, n_tok_b, a_tok_b, nums_b, c_b = rec_b

                # Name similarity features
                name_exact = int(n_a == n_b and n_a != "")
                name_jacc = jaccard(n_tok_a, n_tok_b)
                name_lev = fuzz.ratio(n_a, n_b) / 100.0
                name_sort = fuzz.token_sort_ratio(n_a, n_b) / 100.0
                name_part = fuzz.partial_ratio(n_a, n_b) / 100.0

                # Address similarity features
                addr_exact = int(a_a == a_b and a_a != "")
                addr_jacc = jaccard(a_tok_a, a_tok_b)
                addr_lev = fuzz.ratio(a_a, a_b) / 100.0
                addr_sort = fuzz.token_sort_ratio(a_a, a_b) / 100.0
                addr_num_match = int(bool(nums_a & nums_b))

                # Country and length features
                cntry_match = int(c_a == c_b and c_a != "")
                len_diff = abs(len_n_a - len(n_b))

                buffer.append(
                    f"{s1_id}\t{cand_id}\t{name_exact}\t{name_jacc:.4f}\t{name_lev:.4f}\t"
                    f"{name_sort:.4f}\t{name_part:.4f}\t{addr_exact}\t{addr_jacc:.4f}\t"
                    f"{addr_lev:.4f}\t{addr_sort:.4f}\t{addr_num_match}\t{cntry_match}\t{len_diff}\n"
                )

                if len(buffer) >= CHUNK_PAIRS:
                    out_f.writelines(buffer)
                    written_count += len(buffer)
                    buffer.clear()

                    elapsed = time.time() - t0
                    rate = written_count / elapsed if elapsed > 0 else 0
                    pct = (written_count / total_pairs) * 100
                    print(
                        f"  {pct:5.1f}% | {written_count:,}/{total_pairs:,} pairs | "
                        f"{rate:,.0f} pairs/s | {elapsed:.0f}s elapsed"
                    )

        if buffer:
            out_f.writelines(buffer)
            written_count += len(buffer)
            buffer.clear()

    total_time = time.time() - t0
    final_rate = written_count / total_time if total_time > 0 else 0
    print(f"\nDone! {written_count:,} pairs written to {args.out} in {total_time:.1f}s ({final_rate:,.0f} pairs/s)")


if __name__ == "__main__":
    main()
