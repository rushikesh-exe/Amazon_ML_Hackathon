"""
Macro-averaged F_0.5 exactly as defined in the challenge problem statement:
computed per Source1 entity, then averaged. A singleton (no true matches)
scores 1.0 if predicted empty, 0.0 if any match is predicted.
"""
import argparse
import pandas as pd


def per_entity_f05(true_ids: set, pred_ids: set) -> float:
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    precision = len(true_ids & pred_ids) / len(pred_ids)
    recall = len(true_ids & pred_ids) / len(true_ids)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def macro_f05(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """pred_df / gt_df: columns source1_entity_id, matched_entity_ids (comma-separated)."""
    pred_map = {
        row["source1_entity_id"]: set(row["matched_entity_ids"].split(","))
        if row["matched_entity_ids"] else set()
        for _, row in pred_df.iterrows()
    }
    scores = []
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        true_ids = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
        pred_ids = pred_map.get(s1_id, set())
        scores.append(per_entity_f05(true_ids, pred_ids))
    return sum(scores) / len(scores) if scores else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                         help="tsv with columns source1_entity_id, matched_entity_ids")
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions, sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    score = macro_f05(pred_df, gt_df)
    print(f"Macro F_0.5: {score:.4f}")
