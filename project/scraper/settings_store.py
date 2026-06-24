# =====================================================================
# SETTINGS - the UI-editable scrape FILTERS (valig actor fields).
# Persisted to settings.json so changes survive restarts.
# =====================================================================
import json
import os
import tempfile

from . import config


def _settings_path():
    """Serverless filesystems (e.g. Vercel) are read-only except for the temp dir;
    write there when hosted, otherwise use the bundled settings.json."""
    if os.environ.get("VERCEL"):
        return os.path.join(tempfile.gettempdir(), "scraper_settings.json")
    return config.SETTINGS_PATH

DEFAULT_FILTERS = {
    "title": "",
    "location": "Israel",          # default country
    "experienceLevel": [],         # list of EXP_LEVELS labels
    "datePosted": "",              # "" = any, else a DATE_OPTIONS label
    "contractType": [],            # list of CONTRACT_TYPES labels
    "remote": "",                  # "" = any, else a REMOTE_OPTIONS label
    "limit": config.DEFAULT_LIMIT,
}


def load_settings():
    data = {}
    # prefer the writable copy (temp on serverless), fall back to the bundled defaults
    for p in (_settings_path(), config.SETTINGS_PATH):
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                break
            except Exception:
                data = {}
    filters = {**DEFAULT_FILTERS, **(data.get("filters") or {})}
    return {"filters": filters}


def _sanitize(filters):
    f = {**DEFAULT_FILTERS, **(filters or {})}
    f["title"] = str(f.get("title") or "").strip()
    f["location"] = str(f.get("location") or "Israel").strip() or "Israel"
    f["experienceLevel"] = [x for x in (f.get("experienceLevel") or []) if x in config.EXP_CODE]
    f["contractType"] = [x for x in (f.get("contractType") or []) if x in config.CONTRACT_CODE]
    f["datePosted"] = f["datePosted"] if f.get("datePosted") in config.DATE_CODE else ""
    f["remote"] = f["remote"] if f.get("remote") in config.REMOTE_CODE else ""
    try:
        f["limit"] = max(1, min(1000, int(f.get("limit") or config.DEFAULT_LIMIT)))
    except (TypeError, ValueError):
        f["limit"] = config.DEFAULT_LIMIT
    return f


def save_filters(filters):
    s = load_settings()
    s["filters"] = _sanitize({**s["filters"], **(filters or {})})
    try:
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except OSError:
        pass   # read-only filesystem (serverless) — settings just won't persist; not fatal
    return s
