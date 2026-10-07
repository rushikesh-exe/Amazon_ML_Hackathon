# Entity Resolution — Model 1: Embedding + Blocking

## Design decisions (confirmed with you)
- top_k = 50 candidates per Source1 row
- Country blocking: exact match, implemented generically (groups by
  whatever string appears in `country` — no hardcoded {US, India}, so
  it works unchanged on France in the test set)
- Validate on a 20k-row sample first before running on full files

## What each file does
- `text_normalize.py` — lowercases, transliterates (Devanagari → Latin
  via unidecode), expands abbreviations (Rd→Road, Pvt→Private, etc.)
- `make_sample.py` — streams the full tsv files in chunks (does NOT
  load them fully into memory — needed for source2/3 at ~5M rows each)
  and builds a 20k-row Source1 sample + its true matches + a random
  negative pool from Source2/3
- `build_embeddings.py` — per source file: normalizes text, encodes
  with `all-MiniLM-L6-v2` (Apache 2.0, 384-dim) for blocking, and
  separately fits/saves char n-gram TF-IDF vectors (2–4 char n-grams)
  for later use as a classifier feature on candidate pairs only —
  not used for retrieval, since sparse cosine at full 5M×5M scale isn't
  tractable, while computing it after blocking (small candidate set)
  is cheap
- `generate_candidates.py` — builds a FAISS `IndexFlatIP` per country
  over Source2+3 combined embeddings (inner product = cosine, since
  embeddings are L2-normalized), retrieves top-k per Source1 row,
  writes `candidate_pairs.tsv`
- `evaluate_blocking.py` — recall@k against ground truth: the recall
  ceiling for whatever matcher/classifier runs on top of this

## Run order (sample validation)
```bash
pip install -r requirements.txt

python3 make_sample.py   # writes sample_data/*.tsv

python3 build_embeddings.py \
  --input sample_data/sample_source1.tsv \
  --out_dir sample_data/source1_emb \
  --save_tfidf_vectorizer sample_data/tfidf_vectorizer.pkl

python3 build_embeddings.py \
  --input sample_data/sample_source2.tsv \
  --out_dir sample_data/source2_emb \
  --tfidf_vectorizer sample_data/tfidf_vectorizer.pkl

python3 build_embeddings.py \
  --input sample_data/sample_source3.tsv \
  --out_dir sample_data/source3_emb \
  --tfidf_vectorizer sample_data/tfidf_vectorizer.pkl

python3 generate_candidates.py \
  --source1_dir sample_data/source1_emb \
  --source2_dir sample_data/source2_emb \
  --source3_dir sample_data/source3_emb \
  --top_k 50 \
  --out sample_data/candidate_pairs.tsv

python3 evaluate_blocking.py \
  --candidates sample_data/candidate_pairs.tsv \
  --ground_truth sample_data/sample_ground_truth.tsv
```

## Verified in this environment
- `make_sample.py`: ran end-to-end on the real 2.2M/5M/5.3M-row files
  (streaming, no OOM) → 20k/53k/55k row sample
- `generate_candidates.py` + `evaluate_blocking.py`: ran end-to-end
  against dummy random vectors (correct output format: no duplicate
  IDs, no duplicate source1 rows, tab-separated, avg candidate list
  size == top_k). Near-zero recall with random vectors is expected —
  it only proves the retrieval/eval *logic* is correct.
- `build_embeddings.py`: **not runnable in this sandbox** — no network
  access to huggingface.co to download `all-MiniLM-L6-v2`. Will work
  on Colab or AWS (both have internet access). Run this first, before
  the other two steps, once you're on Colab/AWS.

## Known limitation to fix before full-scale (2.2M/5M/5.3M rows)
`generate_candidates.py` currently builds one `IndexFlatIP` (exact
brute-force search) per country. At full scale, brute-force search of
2.2M queries against ~10M vectors per country will be slow. Before the
full run, swap `IndexFlatIP` for an approximate index
(`IndexIVFFlat` or `IndexHNSWFlat`) — flag for a follow-up, not done
here since we agreed to validate on the sample first.

## Model 2 — pairwise classifier on top of blocking

Run these only after Model 1 (above) has produced `candidate_pairs.tsv`
with good recall (checked via `evaluate_blocking.py`).

- `pairwise_features.py` — shared feature functions (not run directly)
- `build_pair_features.py` — explodes `candidate_pairs.tsv` into one
  row per (Source1, candidate) pair, computes 8 similarity features
  (embedding cosine, TF-IDF cosine, name/address `token_sort_ratio`,
  name/address token Jaccard, name/address length diff). Pass
  `--ground_truth` to attach labels for training; omit it for inference.
- `train_classifier.py` — trains a LightGBM classifier (MIT-licensed,
  small), splits by Source1 entity (not by pair, to avoid leakage),
  and picks a decision threshold by sweeping against the **official
  macro F_0.5 metric** (not a generic precision/recall proxy)
- `predict_matches.py` — applies the trained model + threshold,
  outputs `matching_results.tsv` in the exact required format
- `evaluate_matching.py` — computes the official macro F_0.5 (usable
  standalone, e.g. to score a held-out split)

```bash
pip install -r requirements.txt   # now includes lightgbm, rapidfuzz, tqdm

# 1. Build labeled features from the candidates + ground truth
python3 build_pair_features.py \
  --candidates sample_data/candidate_pairs.tsv \
  --source1_tsv sample_data/sample_source1.tsv --source1_emb_dir sample_data/source1_emb \
  --source2_tsv sample_data/sample_source2.tsv --source2_emb_dir sample_data/source2_emb \
  --source3_tsv sample_data/sample_source3.tsv --source3_emb_dir sample_data/source3_emb \
  --ground_truth sample_data/sample_ground_truth.tsv \
  --out sample_data/train_features.csv

# 2. Train + tune threshold
python3 train_classifier.py \
  --features sample_data/train_features.csv \
  --model_out sample_data/model2_lgbm.pkl \
  --threshold_out sample_data/model2_threshold.txt

# 3. Build the SAME features again, this time without labels (inference mode)
python3 build_pair_features.py \
  --candidates sample_data/candidate_pairs.tsv \
  --source1_tsv sample_data/sample_source1.tsv --source1_emb_dir sample_data/source1_emb \
  --source2_tsv sample_data/sample_source2.tsv --source2_emb_dir sample_data/source2_emb \
  --source3_tsv sample_data/sample_source3.tsv --source3_emb_dir sample_data/source3_emb \
  --out sample_data/inference_features.csv

# 4. Predict final matches
python3 predict_matches.py \
  --features sample_data/inference_features.csv \
  --model sample_data/model2_lgbm.pkl \
  --threshold "$(cat sample_data/model2_threshold.txt)" \
  --all_source1_ids sample_data/sample_source1.tsv \
  --out sample_data/matching_results.tsv

# 5. Score it
python3 evaluate_matching.py \
  --predictions sample_data/matching_results.tsv \
  --ground_truth sample_data/sample_ground_truth.tsv
```

Step 3 recomputing features may look redundant with step 1, but it's
intentional: step 1's output has a `label` column and is only ever
used for training; step 3's output has no labels and is what
`predict_matches.py` scores in the exact form the real test set will
arrive in (no ground truth). Reusing step 1's file for inference would
still work here, but the script is written to match how you'll run it
on the real test set later, where no labeled file will exist at all.

## Verified in this environment (Model 2)
Ran end-to-end on the real sample text with placeholder (non-MiniLM)
embeddings — because this sandbox can't reach huggingface.co, exactly
as with Model 1. Confirmed: feature table builds correctly (8 columns,
1M pair rows for a 20k×50 candidate set), LightGBM trains and the
threshold sweep runs against the real macro F_0.5 implementation,
`predict_matches.py` produces a correctly-formatted `matching_results.tsv`
(no duplicate IDs, one row per Source1 entity), and `evaluate_matching.py`
scores it. Real classifier quality can only be judged once you run this
on Colab with actual MiniLM embeddings.
