"""
Train Model 2: a LightGBM binary classifier on pairwise features
(feature table from build_pair_features.py, which has a 'label' column).

Splits by SOURCE1 ENTITY (not by pair) into train/val, so all candidate
pairs for one Source1 row stay together -- otherwise validation would
leak information about which entity's pairs look like matches.

Threshold is chosen by sweeping candidate probability thresholds and
picking the one that maximizes the OFFICIAL macro F_0.5 metric on the
validation split (not a generic precision/recall/F1 pair-level proxy),
since that's what's actually scored.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from pairwise_features import FEATURE_NAMES
from evaluate_matching import per_entity_f05


def predictions_to_entity_dict(df, id_col, cand_col, prob_col, threshold):
    """Group thresholded positive pairs back into per-entity predicted sets."""
    positive = df[df[prob_col] >= threshold]
    grouped = positive.groupby(id_col)[cand_col].apply(set).to_dict()
    return grouped


def macro_f05_at_threshold(val_df, threshold):
    pred_by_entity = predictions_to_entity_dict(
        val_df, "source1_entity_id", "candidate_entity_id", "pred_prob", threshold
    )
    # every source1 entity that appeared in the candidate table (incl. those
    # with zero surviving predictions above threshold)
    all_entities = val_df.groupby("source1_entity_id")
    true_by_entity = all_entities.apply(
        lambda g: set(g.loc[g["label"] == 1, "candidate_entity_id"])
    ).to_dict()

    scores = []
    for s1_id, true_ids in true_by_entity.items():
        pred_ids = pred_by_entity.get(s1_id, set())
        scores.append(per_entity_f05(true_ids, pred_ids))
    return sum(scores) / len(scores) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Output of build_pair_features.py (with labels)")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--model_out", default="model2_lgbm.pkl")
    parser.add_argument("--threshold_out", default="model2_threshold.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    assert "label" in df.columns, "Feature table has no 'label' column -- rebuild with --ground_truth"

    print(f"Loaded {len(df)} pairs, positive rate {df['label'].mean():.4f}")

    # Group split by source1_entity_id so all of one entity's candidate
    # pairs land entirely in train or entirely in val.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["source1_entity_id"]))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    print(f"Train pairs: {len(train_df)}, Val pairs: {len(val_df)}")

    X_train, y_train = train_df[FEATURE_NAMES], train_df["label"]
    X_val, y_val = val_df[FEATURE_NAMES], val_df["label"]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",  # true matches are a small minority of all candidate pairs
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    val_df["pred_prob"] = model.predict_proba(X_val)[:, 1]

    print("Sweeping thresholds against the official macro F_0.5 metric...")
    # Coarse pass 0.05-0.95, plus a fine pass near the top end (0.90-0.995).
    # The coarse-only sweep found F_0.5 still rising at threshold=0.95 (its
    # upper bound), meaning the true optimum was never actually located --
    # this widens the search so the reported "best" threshold is real.
    coarse = np.arange(0.05, 0.96, 0.05)
    fine_top = np.arange(0.90, 0.996, 0.005)
    candidate_thresholds = sorted(set(np.round(np.concatenate([coarse, fine_top]), 3)))

    best_threshold, best_score = 0.5, -1.0
    for threshold in candidate_thresholds:
        score = macro_f05_at_threshold(val_df, threshold)
        print(f"  threshold={threshold:.2f} -> macro F_0.5={score:.4f}")
        if score > best_score:
            best_threshold, best_score = threshold, score

    print(f"\nBest threshold: {best_threshold:.2f} (val macro F_0.5={best_score:.4f})")

    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)
    with open(args.threshold_out, "w") as f:
        f.write(str(best_threshold))
    print(f"Saved model to {args.model_out}, threshold to {args.threshold_out}")

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp}")


if __name__ == "__main__":
    main()
