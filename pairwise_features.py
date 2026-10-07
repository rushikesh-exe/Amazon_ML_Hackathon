"""
Pairwise similarity features between a Source1 row and a candidate
Source2/3 row, used to train/run the Model 2 classifier on top of
Model 1's blocking output.

All features are symmetric and rely only on data already produced by
build_embeddings.py (embeddings.npy, tfidf.npz) plus the raw tsv text.
"""
import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz

from text_normalize import normalize_text

FEATURE_NAMES = [
    "emb_cosine",
    "tfidf_cosine",
    "name_token_sort_ratio",
    "addr_token_sort_ratio",
    "name_jaccard",
    "addr_jaccard",
    "name_len_diff",
    "addr_len_diff",
]


class SourceData:
    """Loads one source's raw text + embeddings + tfidf, indexed by entity_id."""

    def __init__(self, tsv_path, emb_dir):
        import pandas as pd
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
        self.id_to_row = {eid: i for i, eid in enumerate(df["entity_id"].values)}
        self.norm_names = [normalize_text(n) for n in df["business_name"]]
        self.norm_addrs = [normalize_text(a) for a in df["business_address"]]
        self.embeddings = np.load(f"{emb_dir}/embeddings.npy")
        self.tfidf = sp.load_npz(f"{emb_dir}/tfidf.npz").tocsr()

        emb_ids = np.load(f"{emb_dir}/entity_ids.npy", allow_pickle=True)
        assert list(emb_ids) == list(df["entity_id"].values), (
            f"Row order mismatch between {tsv_path} and {emb_dir} embeddings"
        )

    def row(self, entity_id):
        return self.id_to_row[entity_id]


def _jaccard(tokens_a, tokens_b):
    a, b = set(tokens_a), set(tokens_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_pair_features(s1_data, cand_data, s1_id, cand_id):
    """Compute the feature vector for one (Source1, candidate) pair."""
    i = s1_data.row(s1_id)
    j = cand_data.row(cand_id)

    emb_cosine = float(np.dot(s1_data.embeddings[i], cand_data.embeddings[j]))

    tfidf_cosine = float(s1_data.tfidf[i].multiply(cand_data.tfidf[j]).sum())

    name_a, name_b = s1_data.norm_names[i], cand_data.norm_names[j]
    addr_a, addr_b = s1_data.norm_addrs[i], cand_data.norm_addrs[j]

    name_ratio = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    addr_ratio = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0

    name_jac = _jaccard(name_a.split(), name_b.split())
    addr_jac = _jaccard(addr_a.split(), addr_b.split())

    name_len_diff = abs(len(name_a) - len(name_b))
    addr_len_diff = abs(len(addr_a) - len(addr_b))

    return [
        emb_cosine, tfidf_cosine, name_ratio, addr_ratio,
        name_jac, addr_jac, name_len_diff, addr_len_diff,
    ]
