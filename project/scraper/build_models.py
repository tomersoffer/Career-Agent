# =====================================================================
# BUILD MODELS - trains the scrape-side ML on the master CSV:
#   * Similarity (SEARCH): TF-IDF on TITLE and on LOCATION.
#   * Clustering: KMeans over MEANINGFUL job-market features -
#       job FUNCTION (work_type) + INDUSTRY (sector), both multi-hot,
#       + DISTRICT (one-hot) + EXPERIENCE rank. Title text is deliberately
#       NOT used for clustering (it produced one giant blob); it stays for
#       search only. K is chosen by the elbow method (like the gold pipeline).
#   Cluster ids + derived columns are written back to the master CSV.
# =====================================================================
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import clean, config
from .store import write_csv


def _elbow_k(X, k_min=3, k_max=10):
    """Pick K by the point of maximum distance from the line joining the
    first and last (K, inertia) points - the classic elbow heuristic."""
    ks = list(range(k_min, min(k_max, len(X) - 1) + 1))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in ks]
    x, y = np.array(ks, float), np.array(inertias, float)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return int(ks[int(np.argmax(dist))]), ks, [float(v) for v in inertias]


def build():
    if not os.path.exists(config.MASTER_CSV):
        return {"ok": False, "error": "No scraped data to train on."}
    df = pd.read_csv(config.MASTER_CSV)
    n = len(df)
    if n == 0:
        return {"ok": False, "error": "Master CSV is empty."}

    df = clean.enrich(df)

    # ---- SEARCH vectorizers (title + location) ----
    title_vec = TfidfVectorizer(stop_words="english", min_df=1)
    loc_vec = TfidfVectorizer(min_df=1)
    title_vec.fit(df["title"].fillna("").astype(str).str.lower())
    loc_vec.fit(df["location"].fillna("").astype(str).str.lower())

    # ---- CLUSTER features: function + industry (multi-hot) + district + experience ----
    cluster_ids = np.zeros(n, dtype=int)
    distances = np.zeros(n, dtype=float)
    is_anomaly = np.zeros(n, dtype=int)
    kmeans, chosen_k, sizes = None, 0, {}
    if n >= 8:
        md = max(5, n // 200)   # drop tags rarer than this
        func = CountVectorizer(analyzer=clean.comma_terms, binary=True, min_df=md)
        ind = CountVectorizer(analyzer=clean.comma_terms, binary=True, min_df=md)
        F = func.fit_transform(df["work_type"].fillna("")).toarray().astype(float)
        I = ind.fit_transform(df["sector"].fillna("")).toarray().astype(float)
        D = pd.get_dummies(df["job_district"].fillna("")).to_numpy().astype(float) * 0.5   # geography weighted lower
        E = (df["experience_rank"].to_numpy().reshape(-1, 1).astype(float) / 6.0) * 0.5    # seniority weighted lower
        X = np.hstack([F, I, D, E])

        chosen_k, _, _ = _elbow_k(X, 3, 10)
        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
        cluster_ids = kmeans.labels_
        sizes = {int(k): int(v) for k, v in pd.Series(cluster_ids).value_counts().sort_index().items()}

        # ---- ANOMALY DETECTION: Euclidean distance from assigned centroid, 95th-pct cutoff
        #      (same method as the gold pipeline) -> flags unusual job profiles in the market.
        distances = np.linalg.norm(X - kmeans.cluster_centers_[cluster_ids], axis=1)
        threshold = float(np.percentile(distances, 95))
        is_anomaly = (distances > threshold).astype(int)

    df["cluster_id"] = cluster_ids
    df["centroid_distance"] = distances
    df["is_anomaly"] = is_anomaly
    write_csv(df, config.MASTER_CSV)

    joblib.dump(title_vec, config.TITLE_VEC_PATH)
    joblib.dump(loc_vec, config.LOC_VEC_PATH)
    if kmeans is not None:
        joblib.dump(kmeans, config.KMEANS_PATH)

    return {"ok": True, "rows": n, "clusters": int(len(set(cluster_ids))),
            "k": chosen_k, "sizes": sizes, "anomalies": int(is_anomaly.sum())}
