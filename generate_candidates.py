"""
Generate candidate_pairs.tsv: for each Source1 row, the top-k nearest
Source2/3 rows by MiniLM cosine similarity, blocked by exact country
match.

Country blocking is implemented generically (group by whatever string
value appears in the `country` column) -- NOT hardcoded to {US, India}.
This means it works unchanged on the test set's France rows too, and on
any other country label that shows up, per the problem statement's
"treat country as an open set" requirement.
"""
import argparse
import os
import numpy as np
import pandas as pd
import faiss

TOP_K = 50


def load_source(dir_path):
    return {
        "entity_ids": np.load(f"{dir_path}/entity_ids.npy", allow_pickle=True),
        "embeddings": np.load(f"{dir_path}/embeddings.npy"),
        "countries": np.load(f"{dir_path}/countries.npy", allow_pickle=True),
    }


def build_country_indexes(entity_ids, embeddings, countries):
    """One FAISS flat inner-product index per country value (embeddings are
    already L2-normalized at encode time, so inner product == cosine sim)."""
    indexes = {}
    for country in np.unique(countries):
        mask = countries == country
        vecs = embeddings[mask]
        ids = entity_ids[mask]
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        indexes[country] = {"index": index, "ids": ids}
    return indexes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1_dir", required=True)
    parser.add_argument("--source2_dir", required=True)
    parser.add_argument("--source3_dir", required=True)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--out", required=True, help="Output path for candidate_pairs.tsv")
    args = parser.parse_args()

    print("Loading embeddings...")
    s1 = load_source(args.source1_dir)
    s2 = load_source(args.source2_dir)
    s3 = load_source(args.source3_dir)

    print("Building per-country FAISS indexes for Source2 + Source3 combined...")
    combined_ids = np.concatenate([s2["entity_ids"], s3["entity_ids"]])
    combined_embeddings = np.concatenate([s2["embeddings"], s3["embeddings"]], axis=0)
    combined_countries = np.concatenate([s2["countries"], s3["countries"]])
    indexes = build_country_indexes(combined_ids, combined_embeddings, combined_countries)
    print(f"  countries found in Source2+3: {sorted(indexes.keys())}")

    print(f"Querying top-{args.top_k} candidates per Source1 row...")
    rows = []
    unmatched_country_count = 0
    for i in range(len(s1["entity_ids"])):
        s1_id = s1["entity_ids"][i]
        s1_country = s1["countries"][i]
        s1_vec = s1["embeddings"][i:i + 1]

        if s1_country not in indexes:
            # No Source2/3 rows exist for this country at all -> no candidates.
            unmatched_country_count += 1
            rows.append((s1_id, ""))
            continue

        entry = indexes[s1_country]
        k = min(args.top_k, entry["index"].ntotal)
        _, neighbor_positions = entry["index"].search(s1_vec, k)
        candidate_ids = entry["ids"][neighbor_positions[0]]
        rows.append((s1_id, ",".join(candidate_ids)))

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)

    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Source1 rows with no same-country candidates at all: {unmatched_country_count}")


if __name__ == "__main__":
    main()
