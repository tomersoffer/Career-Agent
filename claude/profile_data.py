"""
TEMPORARY data-profiling script (Step 1 / brainstorming only).
Profiles all 11 CSVs in the LinkedIn Job Postings dataset.
No merging, no output files are produced.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

BASE = r"C:\Users\tomer\OneDrive - The Academic College of Tel-Aviv Jaffa - MTA\Desktop\Agent\Job Posting Dataset"

# logical name -> relative path
FILES = {
    "postings.csv":            "postings.csv",
    "companies.csv":           r"companies\companies.csv",
    "company_industries.csv":  r"companies\company_industries.csv",
    "company_specialities.csv":r"companies\company_specialities.csv",
    "employee_counts.csv":     r"companies\employee_counts.csv",
    "benefits.csv":            r"jobs\benefits.csv",
    "job_industries.csv":      r"jobs\job_industries.csv",
    "job_skills.csv":          r"jobs\job_skills.csv",
    "salaries.csv":            r"jobs\salaries.csv",
    "industries.csv":          r"mappings\industries.csv",
    "skills.csv":              r"mappings\skills.csv",
}


def fmt(x):
    if isinstance(x, float):
        if np.isnan(x):
            return "NaN"
        return f"{x:,.2f}"
    return f"{x:,}" if isinstance(x, (int, np.integer)) else str(x)


def outlier_count(series):
    """Count values beyond 1.5*IQR fences."""
    s = series.dropna()
    if len(s) < 4:
        return 0, None, None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((s < lo) | (s > hi)).sum())
    return n, lo, hi


def profile(name, path):
    full = os.path.join(BASE, path)
    size_mb = os.path.getsize(full) / 1024 / 1024
    print("=" * 100)
    print(f"FILE: {name}    ({size_mb:,.1f} MB)")
    print("=" * 100)

    # Read full file. postings is large but ~500MB fits; use low_memory=False for clean dtypes.
    df = pd.read_csv(full, low_memory=False)

    n_rows = len(df)
    print(f"Total rows: {n_rows:,}   |   Total columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 100)
    header = f"{'column':<28}{'dtype':<12}{'nulls':>12}{'null %':>9}{'unique':>12}"
    print(header)
    print("-" * 100)

    numeric_cols = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        null_pct = 100 * nulls / n_rows if n_rows else 0
        nuniq = int(s.nunique(dropna=True))
        print(f"{col[:27]:<28}{str(s.dtype):<12}{nulls:>12,}{null_pct:>8.1f}%{nuniq:>12,}")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric_cols.append(col)

    if numeric_cols:
        print("-" * 100)
        print("NUMERIC column statistics:")
        nh = f"{'column':<28}{'min':>16}{'max':>18}{'mean':>18}{'outliers(1.5IQR)':>20}"
        print(nh)
        for col in numeric_cols:
            s = df[col]
            n_out, lo, hi = outlier_count(s)
            mn = s.min(); mx = s.max(); mean = s.mean()
            print(f"{col[:27]:<28}{fmt(mn):>16}{fmt(mx):>18}{fmt(mean):>18}{fmt(n_out):>20}")

    # sample of first 2 rows for context (truncated)
    print("-" * 100)
    print("Sample (first 2 rows, values truncated to 60 chars):")
    with pd.option_context("display.max_colwidth", 60):
        print(df.head(2).to_string())
    print()
    del df


if __name__ == "__main__":
    for name, path in FILES.items():
        try:
            profile(name, path)
        except Exception as e:
            print(f"!! ERROR profiling {name}: {e}\n")
