# Smart Career Navigator — Code Submission

This folder contains everything needed to **run the web app** and to **reproduce the full
data-cleaning + ML pipeline**, plus the raw "before" dataset and the cleaned "after" dataset
so the data preparation can be inspected end to end.

---

## Folder layout

```
Career Agent/
├── README.md                     ← this file
├── requirements.txt              ← pinned Python dependencies
├── local.env.example             ← copy to local.env and add your API key
│
├── Job Posting Dataset/          ← BEFORE: raw LinkedIn dataset (11 CSVs, ~531 MB)  [local only*]
│   ├── postings.csv              ← main raw table (493 MB)
│   └── companies/  jobs/  mappings/
│
├── before_vs_after/              ← quick side-by-side (1,000-row samples, open in Excel)
│   ├── BEFORE_postings_raw_sample.csv   (31 messy raw columns)
│   └── AFTER_gold_cleaned_sample.csv    (16 clean, encoded columns)
│
├── benchmarking/                 ← model-comparison & validation TOOLING
│   ├── run_traces.py / run_traces_agent.py        ← produce the trace/eval JSON
│   └── generate_hebrew_doc.py / generate_testing_doc.py  ← build the Hebrew reports
│
├── output/                       ← benchmark RESULTS + validation reports
│   ├── traces.json / traces_agent.json / tool_eval.json / analysis_summary.json
│   └── GPT5.4-mini_Testing_Validation_v2.docx
│
└── project/                      ← all application + pipeline code
    ├── preperation_and_merge.py      ← Phase 2: cleaning & merge  (raw → master)
    ├── analysis_and_profiling.py     ← EDA / data profiling
    ├── training_and_clustering.py    ← Phase 3: K-Means + TF-IDF   (master → gold)
    ├── app.py                        ← Flask web server (entry point)
    ├── agent_runner.py / agent_loop.py / llm_backend.py
    ├── matching.py / analytics.py / build_explainer.py
    ├── cv/  (+ cv/tests/)            ← CV parsing, tailoring, interview flow + unit tests
    ├── scraper/                      ← live Israeli-jobs scraper + bundled seed data
    ├── frontend/                     ← HTML / JS / CSS UI
    ├── tests/                        ← unit tests + model-comparison/eval scripts
    │   ├── test_*.py                          ← pytest unit/integration tests
    │   ├── compare_models*.py / compare_conversation.py   ← Claude vs GPT benchmarks
    │   └── eval_*.py                          ← labeled test cases + accuracy/stability evals
    └── output/                       ← AFTER + trained models (what the app loads)
        ├── gold_linkedin_with_clusters.csv.gz   ← cleaned + clustered dataset (7 MB)
        ├── gold_sample.csv                       ← readable 1,000-row preview
        ├── tfidf_vectorizer.joblib / kmeans_model.joblib / feature_scaler.joblib
        ├── data_dictionary.json
        ├── model_comparison*.json                ← benchmark results
        └── master_jobs_dataset.csv               ← full cleaned master (890 MB)  [local only*]
```

> \* **`[local only]`** = present in this folder but **not in the GitHub repo** (exceeds GitHub's
> 100 MB file limit; also `local.env` is excluded for security). If you cloned this from GitHub,
> the raw dataset and the full 890 MB master are absent — the app still runs (it uses the 7 MB
> `gold...csv.gz`), and the before/after is still visible via the samples in `before_vs_after/`.

---

## Before → After (the data cleaning)

| | File | Rows × Cols |
|---|---|---|
| **BEFORE** (raw) | `Job Posting Dataset/postings.csv` (+ 10 side tables) | 124k × 31 |
| **AFTER** (cleaned + clustered) | `project/output/gold_linkedin_with_clusters.csv.gz` | full × 16 |

The cleaning (in `preperation_and_merge.py`) aggregates the 11 raw tables to one row per job,
applies the smart salary-recovery rules, drops junk / foreign / sub-minimum-wage / duplicate
rows, builds the ML `text_blob`, and encodes the ordinal features. `training_and_clustering.py`
then adds K-Means cluster IDs and anomaly flags. For a quick look, compare the two CSVs in
`before_vs_after/`.

---

## How to run the app

> **Requires Python 3.12.** The dependency versions in `requirements.txt` are pinned
> (`scikit-learn==1.5.1`, `numpy==1.26.4`) because the bundled `.joblib` models were
> pickled with them — a different major version can silently produce wrong results.
> A clean virtual environment is strongly recommended.

```bash
# 1. create + activate a venv (Python 3.12)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. start the web server
cd project
python app.py
# open http://localhost:5000
```

That's it — **no key setup, no pipeline rerun**. A ready-to-use `local.env` with API keys is
already included (see warning below), and the app boots from the bundled `output/` artifacts.

> ⚠️ **`local.env` contains live API keys.** This folder is meant for private hand-off only —
> **do NOT upload it publicly or as a graded submission.** For a public/graded copy, delete
> `local.env` and rename `local.env.example` → `local.env` so each user adds their own key.
> Without a key the app still boots and search works; only the AI advisor is disabled.

## How to reproduce the pipeline (optional)

```bash
cd project
python preperation_and_merge.py     # raw CSVs  → output/master_jobs_dataset.csv
python training_and_clustering.py   # master    → output/gold_linkedin_with_clusters.csv + models
```

Note: re-running regenerates the large uncompressed intermediates
(`master_jobs_dataset.csv`, the full `gold` CSV, `tfidf_matrix.npz`) which were intentionally
**omitted** from the GitHub repo to keep it small — the app does not need them.

---

## Benchmarking & model selection

The model choice (Claude vs GPT-5.4-mini) is backed by real runs, not opinion:

- **`project/tests/compare_models*.py`, `compare_conversation.py`** — run both models over the
  labeled cases in **`project/tests/eval_*.py`** and write **`project/output/model_comparison*.json`**.
- **`benchmarking/run_traces*.py`** — execute the agentic tool-calling loop and write the raw
  traces to **`output/traces*.json`** + **`output/tool_eval.json`** (tool-selection accuracy).
- **`benchmarking/generate_*_doc.py`** — turn those JSON results into the Hebrew reports
  (**`output/GPT5.4-mini_Testing_Validation_v2.docx`**). Every number in the report is sourced
  from a real run — nothing is hand-written.

The **result** files (`*.json`, `.docx`) are committed so you can read the findings without
running anything. To regenerate, the benchmark scripts need a valid `ANTHROPIC_API_KEY` (and
`OPENAI_API_KEY` for the GPT side) in `local.env`, since they call the live models.
