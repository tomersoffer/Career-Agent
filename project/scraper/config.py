# =====================================================================
# SCRAPER CONFIG - paths, constants, and the experience-level mapping.
# The editable LinkedIn URL + default scrape size live in settings.json
# (so the UI can change them); only fixed constants live here.
# =====================================================================
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SETTINGS_PATH = os.path.join(BASE, "settings.json")
USAGE_PATH = os.path.join(BASE, "usage.json")
MASTER_CSV = os.path.join(DATA_DIR, "scraped_jobs.csv")

TITLE_VEC_PATH = os.path.join(MODELS_DIR, "title_vectorizer.joblib")
LOC_VEC_PATH = os.path.join(MODELS_DIR, "location_vectorizer.joblib")
SVD_PATH = os.path.join(MODELS_DIR, "title_svd.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "cluster_scaler.joblib")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans.joblib")

ACTOR_ID = "RIGGeqD6RqKmlVoQU"   # valig "LinkedIn Jobs Scraper" (structured fields, ~$0.0004/result)
DEFAULT_LIMIT = 100              # default jobs per scrape run

# experience_level text -> numeric rank (mirrors the gold dataset's 0..6 scale)
EXP_TEXT_RANK = {
    "internship": 1, "entry level": 2, "associate": 3,
    "mid-senior level": 4, "director": 5, "executive": 6,
}


def exp_rank(value):
    return EXP_TEXT_RANK.get(str(value).strip().lower(), 0)


# ---- field-editor labels -> valig actor codes (single source of truth) --------------
# (exact datePosted/remote/contractType encodings to be confirmed on the first live run)
EXP_CODE = {"Internship": "1", "Entry level": "2", "Associate": "3",
            "Mid-Senior level": "4", "Director": "5", "Executive": "6"}
CONTRACT_CODE = {"Full-time": "F", "Part-time": "P", "Contract": "C",
                 "Temporary": "T", "Internship": "I", "Volunteer": "V", "Other": "O"}
REMOTE_CODE = {"On-site": "1", "Remote": "2", "Hybrid": "3"}
DATE_CODE = {"Past day": "r86400", "Past week": "r604800", "Past month": "r2592000"}

# the option lists the UI renders (label order)
EXP_LEVELS = list(EXP_CODE.keys())
CONTRACT_TYPES = list(CONTRACT_CODE.keys())
REMOTE_OPTIONS = list(REMOTE_CODE.keys())
DATE_OPTIONS = list(DATE_CODE.keys())
