# =====================================================================
# STORE - one accumulating master CSV. Each scrape is UPSERTED: reposts
# (same title+company+location) collapse to the most recently scraped row.
# =====================================================================
import os
import time

import numpy as np
import pandas as pd

from . import config

COLUMNS = ["job_id", "title", "company_name", "company_url", "location", "job_city", "job_district",
           "experience_level", "experience_rank", "contract_type", "work_type", "sector", "salary",
           "recruiter_name", "recruiter_url", "apply_type", "applications_count", "applications_num",
           "posted_at", "posted_time_ago", "job_posting_url", "application_url", "description",
           "scraped_at", "cluster_id", "centroid_distance", "is_anomaly"]


def write_csv(df, path, retries=5):
    """Write atomically (temp + replace) and retry transient locks (OneDrive sync).
    If the file stays locked (e.g. open in Excel) raise a clear, actionable error."""
    tmp = path + ".tmp"
    for attempt in range(retries):
        try:
            df.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise PermissionError(
                    f"לא ניתן לכתוב {os.path.basename(path)} - ייתכן שהקובץ פתוח באקסל. "
                    "סגרו אותו ונסו לסרוק שוב.")
            time.sleep(0.7)


def _key(frame):
    """Dedup primarily by LinkedIn POSTING ID (job_id); fall back to title|company|location
    only for the rare row that has no id. Guarantees no duplicate posting ids."""
    jid = frame["job_id"].astype(str).str.strip().str.lower()
    has_id = ~jid.isin(["", "nan", "none"])
    composite = ("k:" + frame["title"].astype(str).str.strip().str.lower() + "|"
                 + frame["company_name"].astype(str).str.strip().str.lower() + "|"
                 + frame["location"].astype(str).str.strip().str.lower())
    return pd.Series(np.where(has_id, "id:" + jid, composite), index=frame.index)


def upsert(new_rows, scraped_at):
    """Merge freshly scraped rows into the master CSV, keeping the latest per repost key."""
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return count()
    df_new["scraped_at"] = scraped_at
    if "cluster_id" not in df_new:
        df_new["cluster_id"] = -1

    if os.path.exists(config.MASTER_CSV):
        df_old = pd.read_csv(config.MASTER_CSV)
        combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        combined = df_new

    combined["_k"] = _key(combined)
    combined = (combined.sort_values("scraped_at")
                        .drop_duplicates("_k", keep="last")
                        .drop(columns="_k")
                        .reset_index(drop=True))
    _defaults = {"cluster_id": -1, "is_anomaly": 0, "centroid_distance": 0}
    for col in COLUMNS:
        if col not in combined:
            combined[col] = _defaults.get(col, "")
    combined = combined[COLUMNS]
    write_csv(combined, config.MASTER_CSV)
    return len(combined)


def count():
    if not os.path.exists(config.MASTER_CSV):
        return 0
    try:
        return int(len(pd.read_csv(config.MASTER_CSV)))
    except Exception:
        return 0


def last_scraped():
    if not os.path.exists(config.MASTER_CSV):
        return None
    try:
        df = pd.read_csv(config.MASTER_CSV)
        if "scraped_at" in df and len(df):
            return str(df["scraped_at"].max())
    except Exception:
        pass
    return None
