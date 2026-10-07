"""
Evaluate blocking quality: recall@k -- what fraction of true matches
(per ground_truth) were retrieved as candidates. This is the recall
CEILING for whatever final matcher runs on top of these candidates, so
it's the key sanity check on Model 1 before building a classifier on top.
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()

    cand = pd.read_csv(args.candidates, sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)

    cand_map = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))

    total_true_matches = 0
    total_found = 0
    entities_with_matches = 0
    entities_fully_recalled = 0
    candidate_list_sizes = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        true_ids = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
        cand_str = cand_map.get(s1_id, "")
        cand_ids = set(cand_str.split(",")) if cand_str else set()
        candidate_list_sizes.append(len(cand_ids))

        if true_ids:
            entities_with_matches += 1
            found = true_ids & cand_ids
            total_true_matches += len(true_ids)
            total_found += len(found)
            if found == true_ids:
                entities_fully_recalled += 1

    recall = total_found / total_true_matches if total_true_matches else float("nan")
    full_recall_rate = entities_fully_recalled / entities_with_matches if entities_with_matches else float("nan")
    avg_candidates = sum(candidate_list_sizes) / len(candidate_list_sizes) if candidate_list_sizes else 0

    print(f"Source1 entities evaluated: {len(gt)}")
    print(f"Source1 entities with >=1 true match: {entities_with_matches}")
    print(f"Micro recall@k (true matches found / total true matches): {recall:.4f}")
    print(f"Entities with ALL true matches recalled: {full_recall_rate:.4f}")
    print(f"Average candidate list size: {avg_candidates:.1f}")


if __name__ == "__main__":
    main()
