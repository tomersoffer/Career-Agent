# =====================================================================
# SMART CAREER NAVIGATOR - PHASE 2: PREPARATION & MERGE PIPELINE
# ---------------------------------------------------------------------
# Builds the unified one-row-per-job Master Table from the 11 raw CSVs:
#   - aggregates every 1-to-many child table to one-row-per-key BEFORE merge
#   - applies the smart pay_period salary recovery rules
#   - filters junk / duplicate / foreign / sub-minimum-wage rows
#   - compiles text_blob for the similarity model
#   - exports data_dictionary.json and the master CSVs
#   - uses validate='m:1' on every join to guarantee no row explosion
# =====================================================================

import os
import re
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "Job Posting Dataset")
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
                "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI "
                "WY DC PR".split())

EXPERIENCE_RANK = {
    "Internship": 1, "Entry level": 2, "Associate": 3,
    "Mid-Senior level": 4, "Director": 5, "Executive": 6,
}

MIN_WAGE_ANNUAL = 15080   # US federal minimum wage x 2080 hours


def load(rel):
    return pd.read_csv(os.path.join(DATA_DIR, rel), low_memory=False)


# =====================================================================
# SECTION 1 - LOAD RAW TABLES
# =====================================================================

postings = load("postings.csv")
companies = load(os.path.join("companies", "companies.csv"))
company_industries = load(os.path.join("companies", "company_industries.csv"))
employee_counts = load(os.path.join("companies", "employee_counts.csv"))
job_skills = load(os.path.join("jobs", "job_skills.csv"))
job_industries = load(os.path.join("jobs", "job_industries.csv"))
skills_map = load(os.path.join("mappings", "skills.csv"))
industries_map = load(os.path.join("mappings", "industries.csv"))

print("Loaded raw postings rows:", len(postings))
print("------------------------------")

# =====================================================================
# SECTION 2 - AGGREGATE JOB-SIDE CHILD TABLES (1-to-MANY -> 1 per job)
# translate coded skills/industries to names, then collapse per job_id
# =====================================================================

# --- skills: comma list + count ---
js = job_skills.merge(skills_map, on="skill_abr", how="left")
skills_agg = js.groupby("job_id")["skill_name"].agg(
    skills=lambda x: ", ".join(sorted(x.dropna().unique())),
    skill_count="nunique",
).reset_index()
print("Aggregated skills -> rows:", len(skills_agg), "| max skill_count:", int(skills_agg["skill_count"].max()))

# --- job_industry: first NON-BLANK industry name per job ---
ji = job_industries.merge(industries_map, on="industry_id", how="left")
ji_valid = ji[ji["industry_name"].notna()]
jobind_agg = ji_valid.groupby("job_id", sort=False)["industry_name"].first().reset_index()
jobind_agg = jobind_agg.rename(columns={"industry_name": "job_industry"})
print("Aggregated job_industry -> rows:", len(jobind_agg))
print("------------------------------")

# =====================================================================
# SECTION 3 - AGGREGATE COMPANY-SIDE CHILD TABLES (1-to-MANY -> 1 per company)
# =====================================================================

# --- company_industry: first per company ---
compind_agg = (company_industries.drop_duplicates("company_id", keep="first")
               .rename(columns={"industry": "company_industry"}))
print("Aggregated company_industry -> rows:", len(compind_agg))

# --- employee_count: latest snapshot per company ---
emp_agg = (employee_counts.sort_values("time_recorded")
           .drop_duplicates("company_id", keep="last")[["company_id", "employee_count"]])
print("Aggregated employee_count (latest snapshot) -> rows:", len(emp_agg))
print("------------------------------")

# =====================================================================
# SECTION 4 - CLEAN POSTINGS BASE COLUMNS
# title / experience / remote / location parse / posting_date
# =====================================================================

df = postings.copy()

# title: strip whitespace and collapse newlines/tabs
df["title"] = (df["title"].astype(str)
               .str.replace(r"[\n\t]+", " ", regex=True)
               .str.strip())

# experience: fill blanks + ordinal rank
df["experience_level"] = df["formatted_experience_level"].fillna("Not Specified")
df["experience_rank"] = df["experience_level"].map(EXPERIENCE_RANK).fillna(0).astype(int)

# remote flag
df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(int)

# work_type readable (drop CAPS duplicate later)
df["work_type"] = df["formatted_work_type"]

# location parse: City, ST  (the dominant ~85% pattern)
loc_parts = df["location"].astype(str).str.extract(r"^(?P<city>[^,]+),\s*(?P<st>[A-Z]{2})$")
df["job_city"] = loc_parts["city"].str.strip()
df["job_state"] = loc_parts["st"].where(loc_parts["st"].isin(US_STATES))

# posting date from epoch-ms
df["posting_date"] = pd.to_datetime(df["listed_time"], unit="ms").dt.date

print("Missing values check after base cleaning:")
print("  experience_level :", int(df["experience_level"].isna().sum()))
print("  job_state parsed :", int(df["job_state"].notna().sum()), "of", len(df))
print("------------------------------")

# =====================================================================
# SECTION 5 - SMART SALARY RECOVERY (pay_period mislabel correction)
# root cause: wrong pay_period inflates/deflates normalized_salary.
# rules applied in order (see Phase-1 documentation).
# =====================================================================

raw_base = df["med_salary"].fillna((df["min_salary"] + df["max_salary"]) / 2)
salary = df["normalized_salary"].copy()
pay = df["pay_period"]

# rule 1: result > 500k but raw is a plausible annual (10k-500k) -> raw was annual
m1 = (df["normalized_salary"] > 500000) & (raw_base.between(10000, 500000))
salary[m1] = raw_base[m1]

# rule 2: YEARLY raw < 50 -> hourly wage -> x2080
m2 = (pay == "YEARLY") & (raw_base > 1) & (raw_base < 50)
salary[m2] = raw_base[m2] * 2080

# rule 3: YEARLY raw 50-999 -> value entered in thousands -> x1000
m3 = (pay == "YEARLY") & (raw_base >= 50) & (raw_base < 1000)
salary[m3] = raw_base[m3] * 1000

# rule 4: MONTHLY raw < 500 -> hourly mislabel -> x2080
m4 = (pay == "MONTHLY") & (raw_base < 500)
salary[m4] = raw_base[m4] * 2080

# rule 5: HOURLY with corrupt max (max>500 & min<=250) -> use valid min x2080
m5 = (pay == "HOURLY") & (df["max_salary"] > 500) & (df["min_salary"] <= 250)
salary[m5] = df["min_salary"][m5] * 2080

# rule 6: unpaid / placeholder (raw <= 1) -> blank
m6 = raw_base <= 1
salary[m6] = np.nan

df["salary"] = salary

print("Salary recovery applied:")
print(f"  rule1 annual-relabel : {int(m1.sum())}")
print(f"  rule2 yearly<50 x2080: {int(m2.sum())}")
print(f"  rule3 thousands x1000: {int(m3.sum())}")
print(f"  rule4 monthly x2080  : {int(m4.sum())}")
print(f"  rule5 hourly corrupt : {int(m5.sum())}")
print(f"  rule6 unpaid blanked : {int(m6.sum())}")
print("  jobs with salary now :", int(df["salary"].notna().sum()))
print("------------------------------")

# =====================================================================
# SECTION 6 - ROW FILTERING (junk / foreign / sub-min-wage / duplicates)
# =====================================================================

start_rows = len(df)

# (a) junk titles: pure numbers or markdown artifact "!["
junk_mask = df["title"].str.fullmatch(r"\d+") | (df["title"].str.strip() == "![")
df = df[~junk_mask].copy()
print(f"(a) dropped {int(junk_mask.sum())} junk-title rows -> {len(df):,}")

# (b) foreign jobs (USA-only DB) - word-boundary detection on job location
loc = df["location"].astype(str)
is_us = (df["job_state"].notna()
         | loc.str.contains("United States")
         | loc.str.contains(r"Area$|Metroplex|Greater|Bay Area", regex=True))
foreign_re = re.compile(r"\b(Canada|Ontario|Quebec|Philippines|Netherlands|Argentina|"
                        r"Brazil|South Africa|Gambia)\b|,\s(ON|QC|BC)$")
foreign_mask = (~is_us) & loc.apply(lambda x: bool(foreign_re.search(x)))
df = df[~foreign_mask].copy()
print(f"(b) dropped {int(foreign_mask.sum())} foreign-location rows -> {len(df):,}")

# (c) full-time jobs below federal minimum wage = data errors
ftsub_mask = (df["work_type"] == "Full-time") & (df["salary"].notna()) & (df["salary"] < MIN_WAGE_ANNUAL)
df = df[~ftsub_mask].copy()
print(f"(c) dropped {int(ftsub_mask.sum())} full-time sub-min-wage rows -> {len(df):,}")

# (d) duplicate postings: same title+company+location+description -> keep most-viewed
dup_key = (df[["title", "company_name", "location", "description"]]
           .fillna("").astype(str).agg("|".join, axis=1))
df["_dup_key"] = dup_key
df = (df.sort_values("views", ascending=False, na_position="last")
      .drop_duplicates("_dup_key", keep="first")
      .drop(columns="_dup_key"))
df = df.sort_index()
print(f"(d) dropped duplicates -> {len(df):,}")
print(f"TOTAL rows removed: {start_rows - len(df):,}  | final master rows: {len(df):,}")
print("------------------------------")

# =====================================================================
# SECTION 7 - SALARY BAND (numeric 0-7) + CURRENCY NOTE
# all surviving salaries treated as USD (USA-only); band 0 = Not Listed
# =====================================================================

band_conditions = [
    df["salary"].isna(),
    df["salary"] < 40000,
    df["salary"] < 60000,
    df["salary"] < 80000,
    df["salary"] < 100000,
    df["salary"] < 150000,
    df["salary"] < 200000,
]
band_values = [0, 1, 2, 3, 4, 5, 6]
df["salary_band"] = np.select(band_conditions, band_values, default=7).astype(int)

print("salary_band distribution:")
print(df["salary_band"].value_counts().sort_index())
print("------------------------------")

# =====================================================================
# SECTION 8 - MERGE AGGREGATED TABLES (validate to prevent fan-out)
# job-side aggregates are 1:1; company-side aggregates are m:1
# =====================================================================

before = len(df)

df = df.merge(skills_agg, on="job_id", how="left", validate="1:1")
df = df.merge(jobind_agg, on="job_id", how="left", validate="1:1")
df["skill_count"] = df["skill_count"].fillna(0).astype(int)

df = df.merge(companies[["company_id", "name", "company_size"]]
              .rename(columns={"name": "company_name_clean"}),
              on="company_id", how="left", validate="m:1")
df = df.merge(compind_agg[["company_id", "company_industry"]],
              on="company_id", how="left", validate="m:1")
df = df.merge(emp_agg, on="company_id", how="left", validate="m:1")

assert len(df) == before, "ROW COUNT CHANGED DURING MERGE - FAN-OUT DETECTED!"
print("All merges validated (m:1 / 1:1). Row count stable at:", len(df))
print("------------------------------")

# clean company name (prefer companies.csv version), strip whitespace
df["company_name"] = df["company_name_clean"].fillna(df["company_name"])
df["company_name"] = df["company_name"].astype(str).str.strip().replace({"nan": np.nan})

# =====================================================================
# SECTION 9 - COMPANY SIZE (0-7): impute missing from employee_count
# =====================================================================

def emp_to_size(e):
    if pd.isna(e) or e <= 0:
        return np.nan
    if e < 11:    return 1
    if e < 51:    return 2
    if e < 201:   return 3
    if e < 501:   return 4
    if e < 1001:  return 5
    if e < 5001:  return 6
    return 7

imputed = df["employee_count"].apply(emp_to_size)
df["company_size"] = df["company_size"].fillna(imputed).fillna(0).astype(int)

print("company_size distribution (0 = unknown):")
print(df["company_size"].value_counts().sort_index())
print("------------------------------")

# =====================================================================
# SECTION 10 - TEXT BLOB (ML field for TF-IDF / Cosine similarity)
# lowercase, strip URLs/emails/HTML/bad-encoding/punctuation, collapse ws.
# stopword removal is handled later by TfidfVectorizer(stop_words='english').
# =====================================================================

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
HTML_RE = re.compile(r"<[^>]+>")
NONALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(s):
    s = str(s).lower()
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = s.replace(chr(65533), " ")          # bad-encoding replacement char
    s = NONALPHA_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


blob_source = (df["title"].fillna("") + " "
               + df["skills"].fillna("") + " "
               + df["description"].fillna(""))
df["text_blob"] = blob_source.apply(clean_text)
print("text_blob built. Example length stats:")
print(df["text_blob"].str.len().describe().round(0))
print("------------------------------")

# =====================================================================
# SECTION 11 - ACTIONABLE URL META-DATA + ASSEMBLE FINAL MASTER TABLE
# job_posting_url / application_url are retained from postings so the
# agent can route the user directly to the application page. They are
# passive meta-data columns (never used as ML features).
# =====================================================================

print("Application URL columns retained (action-oriented agent):")
print("  job_posting_url non-null :", int(df["job_posting_url"].notna().sum()),
      f"({100 * df['job_posting_url'].notna().mean():.1f}%)")
print("  application_url non-null :", int(df["application_url"].notna().sum()),
      f"({100 * df['application_url'].notna().mean():.1f}%)")
print("------------------------------")

# NOTE: the raw `description` field (which carries embedded HTML markup and is by far the
# heaviest column) is intentionally DROPPED from the exported dataset. Its cleaned,
# HTML-stripped text is already preserved in `text_blob` for the TF-IDF / clustering stage,
# and the agent never reads raw descriptions at runtime - so the master/gold export is far
# lighter with no loss of information needed downstream.
FINAL_COLS = [
    "job_id", "title", "location", "job_posting_url", "application_url",
    "job_city", "job_state", "posting_date",
    "work_type", "experience_level", "remote_allowed", "salary", "salary_band",
    "skills", "skill_count", "job_industry", "company_name", "company_industry",
    "company_size", "experience_rank", "text_blob",
]
master = df[FINAL_COLS].reset_index(drop=True)

print("FINAL MASTER shape:", master.shape)
print("Missing values per final column:")
print(master.isna().sum())
print("------------------------------")

# =====================================================================
# SECTION 12 - DATA DICTIONARY (every encoded/bucketed column)
# =====================================================================

data_dictionary = {
    "salary_band": {
        "0": "Not Listed", "1": "Under $40,000", "2": "$40,000-$60,000",
        "3": "$60,000-$80,000", "4": "$80,000-$100,000", "5": "$100,000-$150,000",
        "6": "$150,000-$200,000", "7": "$200,000+",
    },
    "company_size": {
        "0": "Unknown", "1": "1-10", "2": "11-50", "3": "51-200",
        "4": "201-500", "5": "501-1,000", "6": "1,001-5,000", "7": "5,001+",
    },
    "experience_rank": {
        "0": "Not Specified", "1": "Internship", "2": "Entry level", "3": "Associate",
        "4": "Mid-Senior level", "5": "Director", "6": "Executive",
    },
    "remote_allowed": {"0": "Not flagged as remote", "1": "Remote allowed"},
}
dict_path = os.path.join(OUT_DIR, "data_dictionary.json")
with open(dict_path, "w", encoding="utf-8") as f:
    json.dump(data_dictionary, f, ensure_ascii=False, indent=2)
print("Saved data_dictionary.json")
print("------------------------------")

# =====================================================================
# SECTION 13 - EXPORT MASTER (full + compact), UTF-8
# =====================================================================

full_path = os.path.join(OUT_DIR, "master_jobs_dataset.csv")
compact_path = os.path.join(OUT_DIR, "master_jobs_dataset_compact.csv")

master.to_csv(full_path, index=False, encoding="utf-8-sig")
master.drop(columns=["text_blob"]).to_csv(            # compact = master minus the ML text field
    compact_path, index=False, encoding="utf-8-sig")

print("Saved full master   :", full_path, f"({len(master):,} rows x {master.shape[1]} cols)")
print("Saved compact master:", compact_path)
print("------------------------------")

# =====================================================================
# SECTION 14 - FINAL VALIDATION CHECKS
# =====================================================================

print("VALIDATION:")
print("  unique job_id == rows         :", master["job_id"].is_unique)
print("  salary_band has no nulls      :", int(master["salary_band"].isna().sum()) == 0)
print("  company_size in 0..7          :", master["company_size"].between(0, 7).all())
print("  experience_rank in 0..6       :", master["experience_rank"].between(0, 6).all())
print("  salaried jobs (has salary)    :", int(master["salary"].notna().sum()),
      f"({100 * master['salary'].notna().mean():.1f}%)")
print("------------------------------")
print("PHASE 2 PIPELINE COMPLETE.")
