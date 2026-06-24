# =====================================================================
# SMART CAREER NAVIGATOR - PHASE 3: ML ARCHITECTURE
# ---------------------------------------------------------------------
# Hybrid agent intelligence on the master table:
#   1. K-Means job-profile clustering + distance-based anomaly detection
#   2. TF-IDF Cosine similarity (natural-language role matching)
#   3. Jaccard similarity (skill-set intersection matching)
#   4. Action-oriented hybrid recommender (geo filter -> combined score)
#      that surfaces application URLs and proactive risk alerts.
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import sparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
MASTER_PATH = os.path.join(OUT_DIR, "master_jobs_dataset.csv")

# =====================================================================
# SECTION 1 - LOAD & VALIDATE MASTER DATA
# confirm shape and that the action-oriented URL columns survived
# =====================================================================

df = pd.read_csv(MASTER_PATH, low_memory=False)

print("Master table loaded.")
print("Shape (rows x columns):", df.shape)
print("------------------------------")

print("Action-URL columns present check:")
for col in ["job_posting_url", "application_url"]:
    present = col in df.columns
    non_null = int(df[col].notna().sum()) if present else 0
    print(f"  {col:<18} present={present}  non_null={non_null:,}")
print("------------------------------")

print("Missing values check (ML feature columns):")
for col in ["experience_rank", "company_size", "salary_band", "remote_allowed", "skills"]:
    print(f"  {col:<16} {int(df[col].isna().sum()):>6,} missing")
print("------------------------------")

# =====================================================================
# SECTION 2 - JOB PROFILE CLUSTERING & ANOMALY DETECTION (K-MEANS & DISTANCE)
# cluster jobs by their encoded profile, then flag market anomalies as
# postings that sit unusually far from their own cluster centroid.
# =====================================================================

FEATURES = ["experience_rank", "company_size", "salary_band", "remote_allowed"]
X = df[FEATURES].astype(float).values

# feature scaling (different ordinal ranges -> standardize)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature matrix for clustering:", X_scaled.shape)
print("Features used:", FEATURES)
print("------------------------------")

# --- ELBOW METHOD: evaluate K from 2 to 10 using inertia ---
k_range = range(2, 11)
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"K={k:<2} inertia={km.inertia_:,.0f}")
print("------------------------------")

# objective elbow selection: max distance from the line (first->last point)
ks = np.array(list(k_range))
ys = np.array(inertias)
x1, y1 = ks[0], ys[0]
x2, y2 = ks[-1], ys[-1]
line_dist = np.abs((y2 - y1) * ks - (x2 - x1) * ys + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
optimal_k = int(ks[np.argmax(line_dist)])
print("OPTIMAL K (max distance from elbow line):", optimal_k)
print("------------------------------")

# plot the inertia (elbow) curve
plt.figure()
plt.plot(list(k_range), inertias, marker="o")
plt.axvline(optimal_k, color="red", linestyle="--", label=f"Chosen K = {optimal_k}")
plt.title("Elbow Method - K-Means Inertia")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.legend()
plt.show()

# --- FIT FINAL K-MEANS ---
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

print("Cluster sizes (jobs per cluster):")
cluster_sizes = df["cluster_id"].value_counts().sort_index()
for cid, cnt in cluster_sizes.items():
    print(f"  cluster {cid}: {cnt:>7,}  ({100 * cnt / len(df):4.1f}%)")
print("------------------------------")

# --- TECHNICAL ANOMALY DETECTION: distance from assigned centroid ---
centroids = kmeans.cluster_centers_
assigned_centroids = centroids[df["cluster_id"].values]
distances = np.linalg.norm(X_scaled - assigned_centroids, axis=1)
df["centroid_distance"] = distances

# anomaly threshold = 95th percentile of all centroid distances
anomaly_threshold = np.percentile(distances, 95)
df["is_anomaly"] = (distances > anomaly_threshold).astype(int)

print("ANOMALY DETECTION (distance from cluster centroid):")
print(f"  distance mean = {distances.mean():.3f}  std = {distances.std():.3f}")
print(f"  anomaly threshold (95th percentile) = {anomaly_threshold:.3f}")
print(f"  jobs flagged as anomaly: {int(df['is_anomaly'].sum()):,}  ({100 * df['is_anomaly'].mean():.1f}%)")
print("------------------------------")

# histogram of distances with the anomaly cutoff line
plt.figure()
plt.hist(distances, bins=60, color="steelblue")
plt.axvline(anomaly_threshold, color="red", linestyle="--",
            label=f"Anomaly cutoff (95th pct) = {anomaly_threshold:.2f}")
plt.title("Distribution of Distances from Cluster Centroid")
plt.xlabel("Euclidean distance to assigned centroid (scaled space)")
plt.ylabel("Number of jobs")
plt.legend()
plt.show()

# --- CLUSTER PROFILE SUMMARY (mean of original features per cluster) ---
profile = df.groupby("cluster_id").agg(
    jobs=("job_id", "count"),
    avg_experience_rank=("experience_rank", "mean"),
    avg_company_size=("company_size", "mean"),
    avg_salary_band=("salary_band", "mean"),
    pct_remote=("remote_allowed", "mean"),
    pct_anomaly=("is_anomaly", "mean"),
).round(2)
print("CLUSTER PROFILE SUMMARY (market patterns):")
print(profile)
print("------------------------------")

# scatter: experience vs salary_band colored by cluster (profile separation)
plt.figure()
plt.scatter(df["experience_rank"], df["salary_band"], c=df["cluster_id"],
            cmap="tab10", alpha=0.25, s=8)
plt.title("Job Profiles by Cluster (Experience vs Salary Band)")
plt.xlabel("experience_rank (0=Not Specified ... 6=Executive)")
plt.ylabel("salary_band (0=Not Listed ... 7=$200K+)")
plt.colorbar(label="cluster_id")
plt.show()

# =====================================================================
# SECTION 3 - SIMILARITY MODELS & ACTION-ORIENTED RECOMMENDATION SYSTEM
# Cosine (TF-IDF text) + Jaccard (skill sets) -> hybrid recommender
# =====================================================================

# --- COSINE SIMILARITY: TF-IDF over the engineered text field ---
# text_blob = cleaned(title + skills + description); satisfies the
# title+description text requirement and adds the skills context.
text_corpus = df["text_blob"].fillna("").astype(str)
vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=5)
tfidf_matrix = vectorizer.fit_transform(text_corpus)
print("TF-IDF matrix built:", tfidf_matrix.shape)
print("------------------------------")

# pre-parse each job's skills string into a set (for Jaccard)
job_skill_sets = (df["skills"].fillna("")
                  .apply(lambda s: set(p.strip().lower() for p in s.split(",") if p.strip())))


def jaccard_score(user_set, job_set):
    # |A intersect B| / |A union B|
    if not user_set or not job_set:
        return 0.0
    inter = len(user_set & job_set)
    union = len(user_set | job_set)
    return inter / union if union else 0.0


def get_recommendations(user_query, user_skills, location=None, top_n=5):
    # ---- STEP 1: HARD GEOGRAPHY FILTER ----
    if location:
        mask = (df["job_state"] == location) | (df["job_city"] == location)
        idx = df.index[mask]
    else:
        idx = df.index
    if len(idx) == 0:
        print(f"No jobs found for location '{location}'.")
        return

    # ---- STEP 2: COSINE TEXT SIMILARITY (query vs filtered postings) ----
    query_vec = vectorizer.transform([user_query.lower()])
    cos = cosine_similarity(query_vec, tfidf_matrix[idx]).ravel()

    # ---- STEP 3: JACCARD SKILL SIMILARITY ----
    user_set = set(s.strip().lower() for s in user_skills)
    jac = np.array([jaccard_score(user_set, job_skill_sets.iloc[i]) for i in idx])

    # ---- STEP 4: HYBRID SCORE (50% cosine + 50% jaccard) ----
    combined = 0.5 * cos + 0.5 * jac
    order = np.argsort(combined)[::-1][:top_n]
    top_idx = idx[order]

    print("==============================================")
    print(f"TOP {top_n} RECOMMENDATIONS")
    print(f"Query: '{user_query}'  | Skills: {user_skills}  | Location: {location}")
    print("==============================================")
    for rank, i in enumerate(top_idx, start=1):
        row = df.loc[i]
        score = 0.5 * cos[list(idx).index(i)] + 0.5 * jac[list(idx).index(i)]
        print(f"#{rank}  {row['title']}")
        print(f"    company       : {row['company_name']}  | location: {row['location']}")
        print(f"    hybrid score  : {score:.3f}   (cluster {int(row['cluster_id'])})")
        print(f"    salary_band   : {int(row['salary_band'])}  | experience_rank: {int(row['experience_rank'])}")
        # RISK PROFILING AGENTIC ACTION
        if int(row["is_anomaly"]) == 1:
            print("    ⚠️ AGENT NOTICE: This position is flagged as a market anomaly due to "
                  "an unusual feature combination. Proceed with data verification.")
        # ACTION-ORIENTED OUTPUT: application routing
        print(f"    job_posting_url : {row['job_posting_url']}")
        print(f"    application_url : {row['application_url']}")
        print("    ----------------------------------------")
    print("------------------------------")


# --- DEMO RUN of the action-oriented recommender ---
get_recommendations(
    user_query="data analyst python sql dashboards",
    user_skills={"Information Technology", "Business Development"},
    location="NY",
    top_n=5,
)

# =====================================================================
# SECTION 4 - PIPELINE VERIFICATION & EXPORT
# persist the enriched master + the trained models for the live agent
# =====================================================================

gold_path = os.path.join(OUT_DIR, "gold_linkedin_with_clusters.csv")
df.to_csv(gold_path, index=False, encoding="utf-8-sig")
print("Saved gold master with clusters:", gold_path)
print("  shape:", df.shape, "| new columns: cluster_id, centroid_distance, is_anomaly")

joblib.dump(kmeans, os.path.join(OUT_DIR, "kmeans_model.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))
joblib.dump(vectorizer, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
print("Saved models: kmeans_model.joblib, feature_scaler.joblib, tfidf_vectorizer.joblib")

# precompute & persist the TF-IDF matrix so the agent loads it instantly
# (rows align 1:1 with gold_linkedin_with_clusters.csv) instead of re-vectorizing.
sparse.save_npz(os.path.join(OUT_DIR, "tfidf_matrix.npz"), tfidf_matrix)
print("Saved precomputed TF-IDF matrix: tfidf_matrix.npz", tfidf_matrix.shape)
print("------------------------------")
print("PHASE 3 ML PIPELINE COMPLETE.")
