"""
Build Model 1 embeddings for one source file.

For each row: normalize(business_name) + normalize(business_address) ->
  - a dense MiniLM sentence embedding (used for FAISS candidate retrieval)
  - a char n-gram TF-IDF vector (saved for later use as a classifier
    feature on the *candidate pairs only* -- not used for retrieval,
    since a full 5M x 5M sparse cosine search is not tractable, while
    computing it on the small candidate set after blocking is cheap).

Country is kept as a separate column (used for exact-match blocking in
generate_candidates.py), not folded into the embedded text.

Run once per source file (source1, source2, source3, or their samples).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from text_normalize import build_row_text

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Apache 2.0, 22M params, 384-dim
BATCH_SIZE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a source tsv file")
    parser.add_argument("--out_dir", required=True, help="Directory to save embeddings/artifacts")
    parser.add_argument("--tfidf_vectorizer", default=None,
                         help="Path to a previously-fit TfidfVectorizer pickle. "
                              "If omitted, a new one is fit on THIS file's text "
                              "(fit once on source1, reuse for source2/3 for a shared vocab).")
    parser.add_argument("--save_tfidf_vectorizer", default=None,
                         help="Where to save the fitted vectorizer (only used if fitting fresh).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    print(f"  {len(df)} rows")

    print("Normalizing + concatenating text...")
    texts = [build_row_text(n, a) for n, a in zip(df["business_name"], df["business_address"])]

    print(f"Encoding with {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit-norm -> inner product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    print("Building char n-gram TF-IDF vectors...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    if args.tfidf_vectorizer:
        with open(args.tfidf_vectorizer, "rb") as f:
            vectorizer = pickle.load(f)
        tfidf = vectorizer.transform(texts)
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
        tfidf = vectorizer.fit_transform(texts)
        if args.save_tfidf_vectorizer:
            with open(args.save_tfidf_vectorizer, "wb") as f:
                pickle.dump(vectorizer, f)

    print("Saving artifacts...")
    np.save(f"{args.out_dir}/embeddings.npy", embeddings)
    np.save(f"{args.out_dir}/entity_ids.npy", df["entity_id"].values)
    np.save(f"{args.out_dir}/countries.npy", df["country"].values)
    import scipy.sparse as sp
    sp.save_npz(f"{args.out_dir}/tfidf.npz", tfidf)

    print(f"Done. Embeddings shape: {embeddings.shape}, TF-IDF shape: {tfidf.shape}")


if __name__ == "__main__":
    main()
