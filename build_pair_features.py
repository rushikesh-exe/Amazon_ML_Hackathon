"""
Explode candidate_pairs.tsv into one row per (source1_id, candidate_id)
pair, compute pairwise features, and optionally attach a 0/1 label from
ground truth (omit --ground_truth for inference-time / test-set use,
where no labels exist).
"""
import argparse
import pandas as pd
from tqdm import tqdm

from pairwise_features import SourceData, compute_pair_features, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source1_tsv", required=True)
    parser.add_argument("--source1_emb_dir", required=True)
    parser.add_argument("--source2_tsv", required=True)
    parser.add_argument("--source2_emb_dir", required=True)
    parser.add_argument("--source3_tsv", required=True)
    parser.add_argument("--source3_emb_dir", required=True)
    parser.add_argument("--ground_truth", default=None,
                         help="If given, attach a 0/1 'label' column. Omit at inference time.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("Loading source data (text + embeddings + tfidf)...")
    s1 = SourceData(args.source1_tsv, args.source1_emb_dir)
    s2 = SourceData(args.source2_tsv, args.source2_emb_dir)
    s3 = SourceData(args.source3_tsv, args.source3_emb_dir)

    def get_source(entity_id):
        return s2 if entity_id.startswith("S2-") else s3

    label_map = None
    if args.ground_truth:
        gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
        label_map = {}
        for _, row in gt.iterrows():
            ids = row["matched_entity_ids"]
            label_map[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()

    cand = pd.read_csv(args.candidates, sep="\t", dtype=str, keep_default_na=False)

    records = []
    print("Computing pairwise features...")
    for _, row in tqdm(cand.iterrows(), total=len(cand)):
        s1_id = row["source1_entity_id"]
        cand_ids = row["candidate_entity_ids"]
        if not cand_ids:
            continue
        true_set = label_map.get(s1_id, set()) if label_map is not None else None
        for cand_id in cand_ids.split(","):
            cand_data = get_source(cand_id)
            feats = compute_pair_features(s1, cand_data, s1_id, cand_id)
            rec = {"source1_entity_id": s1_id, "candidate_entity_id": cand_id}
            rec.update(dict(zip(FEATURE_NAMES, feats)))
            if true_set is not None:
                rec["label"] = int(cand_id in true_set)
            records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} pair rows to {args.out}")
    if label_map is not None:
        print(f"Positive rate: {out_df['label'].mean():.4f}")


if __name__ == "__main__":
    main()
