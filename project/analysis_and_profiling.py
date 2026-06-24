# =====================================================================
# SMART CAREER NAVIGATOR - PHASE 1: RAW DATA PROFILING & OUTLIER ANALYSIS
# ---------------------------------------------------------------------
# This script profiles the 11 raw LinkedIn CSV files (BEFORE any cleaning)
# and visualizes baseline distributions / outliers using the 1.5 * IQR rule.
# It produces the "before" picture that motivates every decision made in
# the preparation pipeline (preperation_and_merge.py).
# =====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# data folder is one level up from this script (project/ -> Agent/Job Posting Dataset)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Job Posting Dataset")

FILES = {
    "postings":             "postings.csv",
    "companies":            os.path.join("companies", "companies.csv"),
    "company_industries":   os.path.join("companies", "company_industries.csv"),
    "company_specialities": os.path.join("companies", "company_specialities.csv"),
    "employee_counts":      os.path.join("companies", "employee_counts.csv"),
    "benefits":             os.path.join("jobs", "benefits.csv"),
    "job_industries":       os.path.join("jobs", "job_industries.csv"),
    "job_skills":           os.path.join("jobs", "job_skills.csv"),
    "salaries":             os.path.join("jobs", "salaries.csv"),
    "industries":           os.path.join("mappings", "industries.csv"),
    "skills":               os.path.join("mappings", "skills.csv"),
}


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, FILES[name]), low_memory=False)


def iqr_boxplot(series, title):
    # replicates the annotated-boxplot pattern; adds 1.5*IQR fences + outlier count
    data = series.dropna().values
    if len(data) < 4:
        return
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_outliers = int(((data < lower_fence) | (data > upper_fence)).sum())

    print(f"IQR OUTLIER CHECK: {title}")
    print(f"Q1={q1:.2f}  Median={median:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    print(f"Lower fence (Q1-1.5*IQR)={lower_fence:.2f}  Upper fence (Q3+1.5*IQR)={upper_fence:.2f}")
    print(f"Outliers beyond 1.5*IQR: {n_outliers}  ({100 * n_outliers / len(data):.2f}%)")
    print("------------------------------")

    plt.figure()
    plt.boxplot(data, showmeans=True)
    plt.title(title)
    plt.ylabel(title)
    plt.text(1.05, q1, f"Q1: {q1:.2f}", va="center")
    plt.text(1.05, median, f"Median: {median:.2f}", va="center")
    plt.text(1.05, q3, f"Q3: {q3:.2f}", va="center")
    plt.text(1.05, min_val, f"Min: {min_val:.2f}", va="center")
    plt.text(1.05, max_val, f"Max: {max_val:.2f}", va="center")
    plt.text(1.05, mean_val, f"Mean: {mean_val:.2f}", va="center", color="blue")
    plt.show()


# =====================================================================
# SECTION 1 - LOAD ALL RAW TABLES AND CHECK SHAPES
# inspect how many rows/columns each of the 11 files holds
# =====================================================================

tables = {name: load(name) for name in FILES}

print("RAW TABLE SHAPES (rows x columns):")
for name, df in tables.items():
    print(f"  {name:<22} {df.shape[0]:>8,} x {df.shape[1]}")
print("------------------------------")

postings = tables["postings"]
print("POSTINGS columns:")
print(list(postings.columns))
print("------------------------------")

# =====================================================================
# SECTION 2 - MISSING VALUES CHECK (ALL TABLES)
# isna().sum() per table to expose high-null columns before cleaning
# =====================================================================

for name, df in tables.items():
    print(f"Missing values check: {name}")
    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    if len(na) == 0:
        print("  (no missing values)")
    else:
        for col, cnt in na.items():
            print(f"  {col:<28} {cnt:>8,}  ({100 * cnt / len(df):.1f}%)")
    print("------------------------------")

# =====================================================================
# SECTION 3 - POSTINGS CATEGORICAL DISTRIBUTIONS
# value_counts on the columns that drive Role / Experience / Work-type
# =====================================================================

for col in ["formatted_work_type", "formatted_experience_level", "pay_period",
            "currency", "remote_allowed", "application_type"]:
    print(f"Value counts: {col}")
    print(postings[col].value_counts(dropna=False))
    print("------------------------------")

# =====================================================================
# SECTION 4 - NUMERIC OUTLIER ANALYSIS WITH 1.5 * IQR (BOXPLOTS)
# baseline distributions of the key numeric columns in postings
# =====================================================================

numeric_targets = ["views", "applies", "normalized_salary",
                   "min_salary", "max_salary", "med_salary"]

for col in numeric_targets:
    print(f"DESCRIBE: {col}")
    print(postings[col].describe())
    print("------------------------------")
    iqr_boxplot(postings[col], f"postings.{col} (raw)")

# =====================================================================
# SECTION 5 - SALARY pay_period ANOMALY (THE CORE DATA-QUALITY ISSUE)
# normalized_salary is corrupted: annual values tagged HOURLY get x2080.
# we expose this by comparing the raw typed value vs the normalized value
# per pay_period, and by plotting the inflated outliers.
# =====================================================================

sal = postings[postings["normalized_salary"].notna()].copy()
sal["raw_base"] = sal["med_salary"].fillna((sal["min_salary"] + sal["max_salary"]) / 2)

print("normalized_salary by pay_period (median / max):")
print(sal.groupby("pay_period")["normalized_salary"].agg(["count", "median", "max"]).round(0))
print("------------------------------")

print("raw typed value (raw_base) by pay_period (median / max):")
print(sal.groupby("pay_period")["raw_base"].agg(["median", "max"]).round(0))
print("------------------------------")

# the implied multiplier confirms the formula (HOURLY=2080, MONTHLY=12 ...)
sal["implied_multiplier"] = (sal["normalized_salary"] / sal["raw_base"]).round(1)
print("Implied multiplier (normalized / raw) by pay_period:")
print(sal.groupby("pay_period")["implied_multiplier"].median())
print("------------------------------")

# count how many salaries are physically impossible (the inflated tail)
print("normalized_salary impossible-value counts:")
print("  above $1,000,000 :", int((sal["normalized_salary"] > 1_000_000).sum()))
print("  above $500,000   :", int((sal["normalized_salary"] > 500_000).sum()))
print("  equal to 0 / 1   :", int((sal["raw_base"] <= 1).sum()))
print("------------------------------")

# boxplot of the raw (corrupted) normalized_salary -> shows the extreme skew
iqr_boxplot(sal["normalized_salary"], "normalized_salary RAW (corrupted by pay_period)")

# scatter: raw typed value vs normalized value, colored by pay_period
plt.figure()
period_colors = {"HOURLY": "tab:red", "YEARLY": "tab:blue", "MONTHLY": "tab:green",
                 "WEEKLY": "tab:orange", "BIWEEKLY": "tab:purple"}
for period, color in period_colors.items():
    d = sal[sal["pay_period"] == period]
    plt.scatter(d["raw_base"], d["normalized_salary"], alpha=0.3, label=period, color=color)
plt.title("Raw typed value vs Normalized salary (by pay_period)")
plt.xlabel("Raw typed value (med / avg of min,max)")
plt.ylabel("normalized_salary (annual)")
plt.legend()
plt.show()

# =====================================================================
# SECTION 6 - CHILD TABLE CARDINALITY (1-to-MANY FAN-OUT RISK)
# count rows-per-key for every child table to flag merge explosion risk
# =====================================================================

cardinality_checks = [
    ("companies",            "company_id"),
    ("company_industries",   "company_id"),
    ("employee_counts",      "company_id"),
    ("job_skills",           "job_id"),
    ("job_industries",       "job_id"),
    ("skills",               "skill_abr"),
    ("industries",           "industry_id"),
]

print("ONE-TO-MANY CARDINALITY CHECK (max rows per key):")
for name, key in cardinality_checks:
    df = tables[name]
    vc = df[key].value_counts()
    max_per_key = int(vc.max())
    dup_keys = int((vc > 1).sum())
    risk = "1:1 SAFE" if max_per_key == 1 else f"1:MANY (fan-out x{max_per_key})"
    print(f"  {name:<22} key={key:<12} max/key={max_per_key:<3} dup_keys={dup_keys:<7,} -> {risk}")
print("------------------------------")

# =====================================================================
# SECTION 7 - DUPLICATE POSTINGS BASELINE
# detect repeated postings (same job re-listed) before deduplication
# =====================================================================

dup_key = (postings[["title", "company_name", "location", "description"]]
           .fillna("").astype(str).agg("|".join, axis=1))
n_dup_rows = int(dup_key.duplicated(keep=False).sum())
n_removable = int(len(dup_key) - dup_key.nunique())
print("DUPLICATE POSTINGS (same title+company+location+description):")
print(f"  rows in duplicate groups : {n_dup_rows:,}")
print(f"  removable copies         : {n_removable:,}")
print("------------------------------")

# =====================================================================
# SECTION 8 - GEOGRAPHY BASELINE (USA vs NON-US)
# the agent targets a USA-only market; quantify the foreign tail
# =====================================================================

loc = postings["location"].astype(str)
us_states = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
                "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
                "WV WI WY DC PR".split())
state_abbr = loc.str.extract(r",\s*([A-Z]{2})$")[0]
is_us = (state_abbr.isin(us_states)
         | loc.str.contains("United States")
         | loc.str.contains(r"Area$|Metroplex|Greater|Bay Area", regex=True))
print("GEOGRAPHY BASELINE (by job location):")
print(f"  US-classified jobs     : {int(is_us.sum()):,}  ({100 * is_us.mean():.2f}%)")
print(f"  non-US / unclassified  : {int((~is_us).sum()):,}  ({100 * (~is_us).mean():.2f}%)")
print("------------------------------")

print("PHASE 1 PROFILING COMPLETE.")
