"""
Apply the trained Model 2 classifier + tuned threshold to a (unlabeled)
pairwise feature table, producing matching_results.tsv in the exact
format the challenge requires: one row per Source1 entity, empty string
for no matches, no duplicate ids.
"""
import argparse
import pickle

import pandas as pd

from pairwise_features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Output of build_pair_features.py (no labels needed)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--all_source1_ids", required=True,
                         help="tsv (e.g. sample_source1.tsv) listing every Source1 entity_id "
                              "that must appear in the output, matched or not")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.features)
    df["pred_prob"] = model.predict_proba(df[FEATURE_NAMES])[:, 1]

    positive = df[df["pred_prob"] >= args.threshold]
    matches = positive.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    ).to_dict()

    all_ids = pd.read_csv(args.all_source1_ids, sep="\t", dtype=str)["entity_id"]
    out_df = pd.DataFrame({
        "source1_entity_id": all_ids,
        "matched_entity_ids": [matches.get(eid, "") for eid in all_ids],
    })
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Entities with >=1 predicted match: {(out_df['matched_entity_ids'] != '').sum()}")


if __name__ == "__main__":
    main()
