# =====================================================================
# CLEAN / ENRICH - derive tidy columns from valig's raw fields:
#   location  -> job_city + job_district
#   "84 applicants" -> applications_num
#   experience_level -> experience_rank (0..6)
# and split the compound work_type / sector tags into atomic terms used
# as clustering features.
# =====================================================================
import re

from . import config

DISTRICTS = ["Tel Aviv District", "Center District", "North District",
             "Haifa District", "South District", "Jerusalem District"]


def _district(loc):
    s = str(loc)
    for d in DISTRICTS:
        if d in s:
            return d
    return "Other (Israel)" if "israel" in s.lower() else ""


def _city(loc):
    # "Tel Aviv-Yafo, Tel Aviv District, Israel" -> "Tel Aviv-Yafo"
    parts = [p.strip() for p in str(loc).split(",")]
    head = parts[0] if parts else ""
    if head and "District" not in head and head.lower() != "israel":
        return head
    return ""


def _apps_num(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 0


def comma_terms(v):
    """'Engineering, Finance, and Research' -> ['engineering','finance','research']"""
    s = str(v).lower().replace(" and ", ",")
    return [t.strip() for t in s.split(",") if t.strip() and t.strip() != "nan"]


def enrich(df):
    """Add derived columns (non-destructive). Returns the same df with new columns."""
    df = df.copy()
    df["job_district"] = df["location"].map(_district)
    df["job_city"] = df["location"].map(_city)
    df["applications_num"] = df["applications_count"].map(_apps_num)
    df["experience_rank"] = df["experience_level"].map(config.exp_rank)
    return df
