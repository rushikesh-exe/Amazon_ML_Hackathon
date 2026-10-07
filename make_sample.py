"""
Build a small, self-contained sample to validate the pipeline before
running it on the full ~2.2M / 5M / 5.3M row files.

Streams all files in chunks (does NOT load full files into memory) --
needed here because source2/source3 (~500MB each, ~5M rows) blow past
available RAM as full string DataFrames, and the same streaming
approach is required again at full scale anyway.

Samples N Source1 rows, pulls in every Source2/3 record that is a true
match for them (so recall can be measured), plus a random negative pool
from Source2/3 so the search space isn't trivially small.
"""
import argparse
import os
import pandas as pd
import numpy as np

RNG_SEED = 42
CHUNKSIZE = 200_000


def sample_ids_streaming(path, id_col, n_target, rng):
    """Reservoir-style approximate sampling of id values without loading the full file."""
    with open(path, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    keep_prob = min(1.0, (n_target * 1.5) / max(total_rows, 1))  # oversample slightly, trim later

    kept = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, usecols=[id_col], chunksize=CHUNKSIZE):
        mask = rng.random(len(chunk)) < keep_prob
        kept.append(chunk.loc[mask, id_col])
    ids = pd.concat(kept, ignore_index=True).drop_duplicates()
    if len(ids) > n_target:
        ids = ids.sample(n=n_target, random_state=RNG_SEED)
    return set(ids.tolist())


def extract_rows_streaming(path, id_col, keep_ids):
    """Stream a file, keeping only rows whose id_col is in keep_ids."""
    out_chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNKSIZE):
        out_chunks.append(chunk[chunk[id_col].isin(keep_ids)])
    return pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", default="/mnt/user-data/uploads/train_source1.tsv")
    parser.add_argument("--source2", default="/mnt/user-data/uploads/train_source2.tsv")
    parser.add_argument("--source3", default="/mnt/user-data/uploads/train_source3.tsv")
    parser.add_argument("--ground_truth", default="/mnt/user-data/uploads/train_ground_truth.tsv")
    parser.add_argument("--n_source1", type=int, default=20000)
    parser.add_argument("--n_negatives_per_source", type=int, default=20000)
    parser.add_argument("--out_dir", default="/home/claude/entity_resolution/sample_data")
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    print("Sampling Source1 ids (streaming)...")
    s1_ids = sample_ids_streaming(args.source1, "entity_id", args.n_source1, rng)
    print(f"  -> {len(s1_ids)} Source1 ids selected")

    print("Extracting sampled Source1 rows...")
    s1_sample = extract_rows_streaming(args.source1, "entity_id", s1_ids)

    print("Extracting matching ground-truth rows (streaming)...")
    gt_chunks = []
    for chunk in pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False, chunksize=CHUNKSIZE):
        gt_chunks.append(chunk[chunk["source1_entity_id"].isin(s1_ids)])
    gt_sample = pd.concat(gt_chunks, ignore_index=True) if gt_chunks else pd.DataFrame()

    positive_ids = set()
    for ids in gt_sample["matched_entity_ids"]:
        if ids:
            positive_ids.update(ids.split(","))
    positive_s2_ids = {i for i in positive_ids if i.startswith("S2-")}
    positive_s3_ids = {i for i in positive_ids if i.startswith("S3-")}
    print(f"  -> positive S2 ids: {len(positive_s2_ids)}, positive S3 ids: {len(positive_s3_ids)}")

    print("Sampling negative pool ids from Source2/3 (streaming)...")
    neg_s2_ids = sample_ids_streaming(args.source2, "entity_id", args.n_negatives_per_source, rng)
    neg_s3_ids = sample_ids_streaming(args.source3, "entity_id", args.n_negatives_per_source, rng)

    keep_s2_ids = positive_s2_ids | neg_s2_ids
    keep_s3_ids = positive_s3_ids | neg_s3_ids

    print("Extracting sampled Source2 rows...")
    s2_sample = extract_rows_streaming(args.source2, "entity_id", keep_s2_ids)
    print("Extracting sampled Source3 rows...")
    s3_sample = extract_rows_streaming(args.source3, "entity_id", keep_s3_ids)

    os.makedirs(args.out_dir, exist_ok=True)
    s1_sample.to_csv(f"{args.out_dir}/sample_source1.tsv", sep="\t", index=False)
    s2_sample.to_csv(f"{args.out_dir}/sample_source2.tsv", sep="\t", index=False)
    s3_sample.to_csv(f"{args.out_dir}/sample_source3.tsv", sep="\t", index=False)
    gt_sample.to_csv(f"{args.out_dir}/sample_ground_truth.tsv", sep="\t", index=False)

    print(f"\nSample sizes -> source1: {len(s1_sample)}, source2: {len(s2_sample)}, "
          f"source3: {len(s3_sample)}, gt rows: {len(gt_sample)}")
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
