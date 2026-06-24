# =====================================================================
# BUILD EXPLAINER - dev-only generator for the "How it works" modals.
# Renders themed SVG charts + bakes a single explain.json (prose authored
# here in Hebrew; numbers/sample tables computed from the REAL data).
# Run once locally:  python build_explainer.py
# Output -> frontend/assets/explain/  (static; served with zero runtime cost)
#
# NOTE: uses matplotlib + sklearn + the FULL local data. These are dev-only;
# they are NOT runtime dependencies of the deployed app.
# =====================================================================
import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
SCRAPED = os.path.join(BASE, "scraper", "data", "scraped_jobs.csv")
ASSETS = os.path.join(BASE, "frontend", "assets", "explain")
os.makedirs(ASSETS, exist_ok=True)

# ---- theme (LinkedIn blues) ----
BLUE, BLUE2, BLUE3 = "#0A66C2", "#378FE9", "#70B5F9"
INK, MUTED, GRID, WARN = "#1B1F23", "#62707E", "#E4EBF2", "#B24020"
plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 100,
})

SALARY_LABELS = ["Not listed", "<$40K", "$40-60K", "$60-80K", "$80-100K", "$100-150K", "$150-200K", "$200K+"]
EXP_LABELS = ["Not spec.", "Internship", "Entry", "Associate", "Mid-Senior", "Director", "Executive"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, name):
    path = os.path.join(ASSETS, name)
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("  chart:", name)


def bar(values, labels, title, ylabel="jobs", color=BLUE, rotate=0, fname=None):
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.bar(range(len(values)), values, color=color, width=0.72)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _style(ax)
    _save(fig, fname)


# =====================================================================
# DATASET (global US) charts + facts
# =====================================================================
def build_dataset():
    print("Dataset charts ...")
    cols = ["title", "company_name", "job_state", "experience_level", "experience_rank",
            "company_size", "salary_band", "remote_allowed", "cluster_id",
            "centroid_distance", "is_anomaly"]
    df = pd.read_csv(os.path.join(OUT_DIR, "gold_linkedin_with_clusters.csv"),
                     usecols=cols, low_memory=False)
    n = len(df)

    # salary band distribution
    sb = df["salary_band"].astype(int).value_counts().reindex(range(8), fill_value=0)
    bar(sb.values, SALARY_LABELS, "Salary band distribution", rotate=40, fname="ds_salary.svg")

    # experience distribution
    ex = df["experience_rank"].astype(int).value_counts().reindex(range(7), fill_value=0)
    bar(ex.values, EXP_LABELS, "Experience level distribution", rotate=40, fname="ds_exp.svg")

    # elbow (faithful: saved scaler + KMeans inertia 2..10)
    feats = ["experience_rank", "company_size", "salary_band", "remote_allowed"]
    scaler = joblib.load(os.path.join(OUT_DIR, "feature_scaler.joblib"))
    Xs = scaler.transform(df[feats].astype(float).values)
    ks = list(range(2, 11))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs).inertia_ for k in ks]
    chosen_k = int(df["cluster_id"].nunique())
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.plot(ks, inertias, marker="o", color=BLUE, linewidth=2)
    ax.axvline(chosen_k, color=WARN, linestyle="--", linewidth=1.5, label=f"Chosen K = {chosen_k}")
    ax.set_title("Elbow method — K selection")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "ds_elbow.svg")

    # cluster sizes
    cs = df["cluster_id"].astype(int).value_counts().sort_index()
    bar(cs.values, [f"C{i}" for i in cs.index], "Cluster sizes", fname="ds_clusters.svg")

    # anomaly histogram (stored centroid_distance + 95th pct line)
    dist = df["centroid_distance"].astype(float).values
    thr = float(np.percentile(dist, 95))
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.hist(dist, bins=60, color=BLUE2)
    ax.axvline(thr, color=WARN, linestyle="--", linewidth=1.5, label=f"95th pct = {thr:.2f}")
    ax.set_title("Anomaly detection — distance from centroid")
    ax.set_xlabel("Euclidean distance (scaled)")
    ax.set_ylabel("jobs")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, "ds_anomaly.svg")

    sample = df.head(5).copy()
    sample_rows = [{
        "title": str(r["title"])[:34],
        "company": str(r["company_name"])[:22],
        "state": str(r["job_state"]),
        "experience": str(r["experience_level"]),
        "salary": SALARY_LABELS[int(r["salary_band"])],
        "cluster": int(r["cluster_id"]),
        "anomaly": "⚠" if int(r["is_anomaly"]) == 1 else "",
    } for _, r in sample.iterrows()]

    return {
        "stats": {"jobs": n, "anomalies": int(df["is_anomaly"].sum()),
                  "clusters": chosen_k, "states": int(df["job_state"].nunique())},
        "sample_cols": ["title", "company", "state", "experience", "salary", "cluster", "anomaly"],
        "sample": sample_rows,
    }


# =====================================================================
# SCRAPE (live Israel) charts + facts
# =====================================================================
def build_scrape():
    print("Scrape charts ...")
    from scraper import clean
    df = pd.read_csv(SCRAPED, low_memory=False)
    if "job_district" not in df or df["job_district"].isna().all():
        df = clean.enrich(df)
    n = len(df)

    # district distribution
    dd = df["job_district"].fillna("").replace("", "Other").value_counts().head(7)
    bar(dd.values, [str(x).replace(" District", "") for x in dd.index],
        "Jobs by district", rotate=40, fname="il_district.svg")

    # experience distribution
    el = df["experience_level"].fillna("Not spec.").value_counts().head(6)
    bar(el.values, [str(x).replace(" level", "") for x in el.index],
        "Experience level", rotate=40, fname="il_exp.svg")

    # cluster sizes
    if "cluster_id" in df:
        cs = df["cluster_id"].fillna(-1).astype(int)
        cs = cs[cs >= 0].value_counts().sort_index()
        if len(cs):
            bar(cs.values, [f"C{i}" for i in cs.index], "Cluster sizes", fname="il_clusters.svg")

    # sector mix (multi-value -> atomic terms, top 8)
    terms = {}
    for v in df["sector"].fillna(""):
        for t in clean.comma_terms(v):
            terms[t] = terms.get(t, 0) + 1
    top = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:8]
    if top:
        bar([c for _, c in top], [t[:16] for t, _ in top],
            "Top industries (sector)", rotate=40, color=BLUE2, fname="il_sector.svg")

    # anomaly histogram (centroid distance + 95th-pct cutoff) — same method as the gold pipeline
    anomalies = 0
    if "centroid_distance" in df and df["centroid_distance"].astype(float).gt(0).any():
        dist = df["centroid_distance"].astype(float).values
        thr = float(np.percentile(dist, 95))
        anomalies = int((df["is_anomaly"].astype(float) == 1).sum())
        fig, ax = plt.subplots(figsize=(4.8, 2.9))
        ax.hist(dist, bins=50, color=BLUE2)
        ax.axvline(thr, color=WARN, linestyle="--", linewidth=1.5, label=f"95th pct = {thr:.2f}")
        ax.set_title("Anomaly detection — distance from centroid")
        ax.set_xlabel("Euclidean distance"); ax.set_ylabel("jobs")
        ax.legend(frameon=False); _style(ax); _save(fig, "il_anomaly.svg")

    apps_col = "applications_count" if "applications_count" in df else "applications_num"
    sample = df.head(5).copy()
    sample_rows = [{
        "title": str(r.get("title", ""))[:32],
        "company": str(r.get("company_name", ""))[:20],
        "district": str(r.get("job_district", "")).replace(" District", ""),
        "experience": str(r.get("experience_level", "")),
        "applicants": str(r.get(apps_col, "")),
        "cluster": int(r["cluster_id"]) if "cluster_id" in df and pd.notna(r.get("cluster_id")) else -1,
    } for _, r in sample.iterrows()]

    clusters = int(df["cluster_id"].dropna().nunique()) if "cluster_id" in df else 0
    return {
        "stats": {"jobs": n, "clusters": clusters, "anomalies": anomalies,
                  "districts": int(df["job_district"].replace("", np.nan).nunique())},
        "sample_cols": ["title", "company", "district", "experience", "applicants", "cluster"],
        "sample": sample_rows,
    }


# =====================================================================
# CONTENT (authored Hebrew prose) + assemble explain.json
# =====================================================================
def content(ds_facts, il_facts):
    A = ds_facts["stats"]
    dataset_steps = [
        {"key": "data", "label": "1 · נתונים",
         "summary": f"המאגר מבוסס על LinkedIn Job Postings (2023–2024) מ-Kaggle — 11 קבצי CSV מקושרים. "
                    f"נקודת הפתיחה: 123,849 משרות ו-24,473 חברות; לאחר ניקוי נותרו {A['jobs']:,} משרות בארה\"ב.",
         "points": [
             "מבנה היעד: טבלה אחת מאוחדת (Master Table), שורה אחת לכל משרה (job_id).",
             "מיזוג בטוח: ארבע טבלאות אחד-לרבים (כישורים, תעשיות, מספרי עובדים) צומצמו לשורה אחת למפתח "
             "לפני left-join, עם merge(validate='m:1') ובדיקת מספר שורות — כדי למנוע התפוצצות שורות.",
             "מסד לארה\"ב בלבד — הסינון לפי מיקום המשרה (לא מטה החברה).",
             "הוסרו 4,732 שורות: 4,673 כפילויות (נשמר העותק עם הכי הרבה צפיות), 43 משרות עם שכר מתחת "
             "למינימום, 12 משרות זרות ו-3 כותרות פסולות.",
         ],
         "table": True},
        {"key": "preprocess", "label": "2 · עיבוד מקדים",
         "summary": "כל עמודה נוקתה, קודדה והוכנה למודלים. עיקרון מנחה: לא מוחקים שורות בשל ערכים חסרים — "
                    "חוסר מקודד כדגל/קטגוריה, כדי לשמר את היקף ה-Big Data ולמנוע הטיה באשכול.",
         "points": [
             "שכר → נרמול לשנתי ב-USD + תיקון חכם של pay_period (שכר שעתי/חודשי שתויג כשנתי): כ-420 ערכי "
             "שכר אמיתיים שוחזרו במקום שנמחקו (שעתי×2080, חודשי×12...). נוסף salary_band (0–7).",
             "experience_level → experience_rank מספרי סדור (0–6); company_size → סולם 0–7; "
             "remote_allowed → דגל בינארי.",
             "location → פוצל ל-job_city + job_state (≈85% נפרסים); skills → תרגום קודים לשמות + skill_count.",
             "text_blob → שדה ML מנוקה (title + skills + description): אותיות קטנות, הסרת URL/אימייל/HTML/"
             "קידוד פגום, סינון מילות עצירה — זהו הקלט ל-TF-IDF.",
             "השדה description הגולמי (כולל תגי HTML) הוסר מהתוצר; הטקסט המנוקה נשמר ב-text_blob. "
             "הסכמה הסופית: 21 עמודות, כולן מתועדות ב-data_dictionary.json.",
         ],
         "charts": ["ds_salary.svg", "ds_exp.svg"]},
        {"key": "cluster", "label": "3 · אשכול K-Means",
         "summary": "אלגוריתם K-Means מקבץ את המשרות ל'פרופילי שוק' הומוגניים. ארבעה מאפיינים מקודדים "
                    "נכנסים למודל: experience_rank, company_size, salary_band, remote_allowed — לאחר StandardScaler.",
         "points": [
             "התקנון (StandardScaler) מאזן את השפעת הסקאלות השונות על מדד המרחק.",
             "מספר האשכולות נבחר בשיטת המרפק (Elbow): חושב ה-Inertia (סכום ריבועי המרחקים התוך-אשכוליים) "
             f"עבור K=2..10, ונבחר אובייקטיבית ה-K עם המרחק המרבי מהקו שבין הקצוות → K={A['clusters']}.",
             "האשכולות שהתקבלו מאוזנים (ללא אשכול דומיננטי) ומפרשים מגזרי שוק ברורים.",
             "כל משרה מקבלת cluster_id קבוע; הסגמנט מוצג למשתמש לצד כל המלצה.",
         ],
         "charts": ["ds_elbow.svg", "ds_clusters.svg"]},
        {"key": "anomaly", "label": "4 · זיהוי אנומליות",
         "summary": "על בסיס האשכולות מזוהות 'אנומליות שוק' — משרות עם צירוף מאפיינים חריג "
                    "(למשל שכר גבוה במיוחד לצד דרישת ניסיון אפסית), שעלולות להעיד על מודעה שגויה או חשודה.",
         "points": [
             "לכל משרה חושב המרחק האוקלידי בין וקטור המאפיינים המתוקנן לבין צנטרואיד האשכול שאליו שויכה: "
             "d = √Σ(xᵢ − cᵢ)².",
             f"סף האנומליה = אחוזון 95 של התפלגות המרחקים. סף = 2.13 → {A['anomalies']:,} משרות (5.0%) "
             "סומנו ב-is_anomaly=1.",
             "הדגל מוצג למשתמש כהתראת סיכון יזומה בעת ההמלצה — להגנה מפני מודעות שגויות.",
         ],
         "charts": ["ds_anomaly.svg"]},
        {"key": "match", "label": "5 · מהבקשה למשרות",
         "summary": "בזמן ריצה, הבקשה החופשית שלך הופכת למשרות מדורגות. זהו צינור RAG מקומי בזיכרון "
                    "(In-Memory) — ללא מסד נתונים וקטורי חיצוני. אותו מנוע התאמה (matching.py) משרת את שתי "
                    "הלשוניות.",
         "points": [
             "פענוח LLM-first: המודל מבין את הבקשה ומחלץ תפקיד, מילים נרדפות, עיר, מדינה ושנות ניסיון — "
             "מעוגן למדינות הקיימות בפועל בנתונים (לא ימציא מדינה ריקה).",
             "שלושה ממדים בלתי-תלויים — כותרת, מיקום וניסיון. כל ממד שצוין בבקשה פועל כמסנן AND נפרד; "
             "ממד שלא הוזכר אינו מגביל — כך 'משרות ב-NY' לבד או 'מנהל בכיר' לבד עובדים גם הם.",
             "ציון כותרת = דמיון קוסינוס מרבי-על-וריאנטים מעל TF-IDF: התפקיד וכל מילה נרדפת מנוקדים בנפרד "
             "והגבוה מנצח (נרדפות במשקל 0.9) — כדי שמונח ספציפי לא ייבלע במילים גנריות.",
             "ציון ניסיון = קרבה מספרית בין experience_rank המבוקש לזה של המשרה (1−|Δ|/6), לא דמיון טקסט; "
             "משרות עם ניסיון לא-ידוע (דרגה 0) מקבלות חזקת תמימות ועוברות את השער.",
             "ציון מיקום = מדינת היעד 1.0, מדינות סמוכות פוחת (≈0.62 ומטה לפי קרבה) — פולבק גאוגרפי כציון "
             "רציף, לא מפל קשיח; בקשת 'עבודה מרחוק' מסננת מראש.",
             "הציונים מאוחדים לדירוג היברידי: Score = 0.5×כותרת + 0.3×מיקום + 0.2×ניסיון. המובילים נבחרים — "
             "וכל משרה כבר נושאת את סגמנט האשכול ואת דגל האנומליה שחושבו אופליין בשלבים 3–4.",
         ],
         "diagram": "flow"},
        {"key": "llm", "label": "6 · סוכן LLM ופעולה",
         "summary": "עובדות המשרות הנבחרות מועברות למודל השפה (LLM), שמנסח אינטרו קצר ואישי בעברית "
                    "(משפט–שניים) מעל כרטיסי המשרות — בפורמט JSON מובנה. המספרים וההתאמות מחושבים בקוד; "
                    "ה-LLM רק מנסח.",
         "points": [
             "פרופיל סיכון: למשרה שסומנה כאנומליה הסוכן מצרף '⚠️ התראת סיכון' ומסביר שהפרסום סוטה "
             "סטטיסטית מקו הבסיס של אשכול השוק — ומומלץ לאמתו.",
             "פעולה (Action-Oriented): כל כרטיס משרה מציג את קישורי job_posting_url ו-application_url "
             "להגשה מיידית.",
             "כל חישובי הדמיון מקומיים בזיכרון; מפתח ה-API נטען מ-local.env המוחרג מבקרת גרסאות.",
         ]},
        {"key": "agent", "label": "7 · סוכן שיחה מלא",
         "summary": "מעבר לחיפוש בודד, המערכת היא צ'אטבוט תעסוקתי: באותו חלון שיחה אפשר לחפש משרות, "
                    "לשאול שאלות נתונים, לקבל ייעוץ קריירה, ולהעלות קורות חיים ולהתאים אותם — והסוכן "
                    "מחליט בעצמו מה להפעיל בכל תור.",
         "points": [
             "לולאת כלים אג'נטית: המודל מקבל חמישה כלים (חיפוש משרות, סטטיסטיקה, עוד תוצאות, ייעוץ "
             "קריירה, שיחה קלה) ומחליט בעצמו אילו להפעיל ובאילו ארגומנטים — ויכול לשרשר עד 4 קריאות "
             "לפני שהוא עונה. הכלים רצים מול אותו מנוע דטרמיניסטי (חיפוש/אנליטיקה/ייעוץ).",
             "מנוע אנליטיקה דטרמיניסטי (רשימה לבנה של פונקציות: שכר ממוצע, ספירת משרות, כישורים/מיקומים/"
             "תעשיות נפוצים, שיעור עבודה מרחוק, פילוח ניסיון) עונה על שאלות נתונים — ה-LLM בוחר פונקציה "
             "ופרמטרים, pandas מחשב, וה-LLM רק מנסח את המספרים (ללא eval/הזרקת קוד).",
             "קורות חיים: העלאה מפעילה ראיון מודרך קצר (שאלה אחת בכל תור) שבונה פרופיל, ואז דירוג "
             "מותאם-CV עם אחוז התאמה לכל משרה (קוסינוס תפקיד) ופער-כישורים (הפרש קבוצות דטרמיניסטי).",
             "התאמת קו\"ח שיחתית: הסוכן מוביל מעבר סעיף-אחר-סעיף (תקציר→כישורים→ניסיון), שואל על פער "
             "אחד בכל תור, ומוסיף רק מה שהמשתמש מאשר — ללא המצאת ניסיון שאין.",
             "עיגון ובטיחות: כל המספרים וההתאמות מחושבים ב-Python; ה-LLM רק מנסח ומחליט. כל פלט מוצג "
             "עובר סניטציה (DOMPurify) למניעת XSS מטקסט משרה מצד-ג'.",
         ]},
    ]
    scrape_steps = [
        {"key": "scrape", "label": "1 · סריקה חיה",
         "summary": f"המשרות נשלפות חי מ-LinkedIn (ישראל) דרך Apify — ה-actor של valig. כ-{il_facts['stats']['jobs']:,} "
                    f"משרות נסרקו כבסיס לדמו. זו תוספת חדשנות מעבר לדרישת המטלה.",
         "points": [
             "ה-actor מקבל שדות סינון מובנים: תפקיד/מילת מפתח, מיקום, רמת ניסיון, סוג העסקה, עבודה מרחוק "
             "וטווח תאריכים — ה-LLM יכול לתרגם בקשה חופשית לשדות האלה.",
             "דה-דופ לפי מזהה הפרסום (job_id) — אין כפילויות; כל המשרות בישראל.",
             "הסריקה רצה לפי דרישה ('סרוק עכשיו') או מתוזמנת. בגרסה המתארחת בענן הסריקה החיה מושבתת "
             "(מערכת קבצים לקריאה בלבד) — אך החיפוש במשרות שכבר נסרקו עובד.",
         ],
         "table": True},
        {"key": "clean", "label": "2 · ניקוי והעשרה",
         "summary": "השדות הגולמיים של valig מסודרים ומועשרים לפני המידול — כדי שיהיו ניתנים לאשכול ולחיפוש.",
         "points": [
             "location → job_district (התאמה ל-6 מחוזות ישראל) + job_city.",
             "\"84 applicants\" → applications_num מספרי לצורך מדד תחרותיות.",
             "experience_level → experience_rank; תגי work_type/sector מפוצלים למונחים אטומיים "
             "(פונקציה ותעשייה) עבור פיצ'רים multi-hot.",
         ],
         "charts": ["il_district.svg", "il_exp.svg"]},
        {"key": "cluster", "label": "3 · אשכול",
         "summary": "המשרות מקובצות לפי תחום ותעשייה — ולא רק לפי כותרת — כדי להימנע מאשכול-ענק יחיד "
                    "ולקבל סגמנטים בעלי משמעות.",
         "points": [
             "פיצ'רים multi-hot: work_type (פונקציה) + sector (תעשייה) + מחוז (משקל נמוך) + experience_rank.",
             "K נבחר בשיטת המרפק, בדיוק כמו במאגר הגלובלי. הכותרת (TF-IDF) שמורה לחיפוש בלבד — לא לאשכול.",
             "התוצאה: סגמנטים מאוזנים בהרבה — האשכול הגדול ירד מ-71% ל-44%.",
         ],
         "charts": ["il_clusters.svg", "il_sector.svg"]},
        {"key": "anomaly", "label": "4 · זיהוי אנומליות",
         "summary": "כמו במאגר הגלובלי, גם המשרות הסרוקות עוברות זיהוי אנומליות — איתור פרופילים חריגים "
                    "שיושבים רחוק ממרכז האשכול שלהם.",
         "points": [
             "לכל משרה חושב המרחק האוקלידי מצנטרואיד האשכול שאליו שויכה; הסף = אחוזון 95 של ההתפלגות.",
             f"התוצאה: {il_facts['stats']['anomalies']:,} משרות (כ-5%) סומנו ב-is_anomaly=1 — צירוף חריג של "
             "תפקיד/תעשייה/בכירות/מיקום, שעלול להעיד על מודעה חשודה או על הזדמנות נדירה.",
             "הסוכן מצרף '⚠️ התראת סיכון' למשרה כזו וממליץ לאמת את הפרטים לפני הגשה.",
         ],
         "charts": ["il_anomaly.svg"]},
        {"key": "match", "label": "5 · התאמה — אותו מנוע",
         "summary": "בדיוק אותו מנוע חיפוש כמו במאגר הגלובלי: מודול matching.py משותף לשתי הלשוניות "
                    "(title_scores, experience_scores, ושער AND זהה).",
         "points": [
             "חיפוש לפי כותרת + רמת ניסיון + מיקום, בכל שילוב שתבחר; כל ממד שצוין מסנן בנפרד (שער AND).",
             "אותו ציון היברידי (0.5×כותרת + 0.3×מיקום + 0.2×ניסיון), אותה התאמת-נרדפות מרבית-על-וריאנטים "
             "ואותה קרבת-ניסיון מספרית כמו במאגר הגלובלי.",
             "אותו פענוח LLM-first של הבקשה. רק אסטרטגיית המיקום שונה — כאן דמיון קוסינוס על העיר במקום "
             "סינון מדינה קשיח.",
         ],
         "diagram": "flow"},
        {"key": "llm", "label": "6 · סוכן LLM",
         "summary": "אותו יועץ LLM — כותב אינטרו קצר (משפט–שניים) מעל כרטיסי המשרות שנסרקו, בפורמט "
                    "JSON מובנה.",
         "points": [
             "דגש על רמת התחרותיות (מספר מועמדים), הסגמנט והתחום של כל משרה.",
             "כל כרטיס משרה מציג קישור ישיר להגשת מועמדות.",
             "אותו שכבת שיחה: גם כאן אפשר לקבל ייעוץ קריירה ולהשתמש בקורות החיים — נתוני שכר/תעשייה "
             "דלילים מדי בישראל, ולכן שאלות נתונים מטופלות כייעוץ.",
         ]},
    ]
    return {
        "dataset": {"title": "מאגר גלובלי · משרות בארה\"ב", "stats": A,
                    "sample_cols": ds_facts["sample_cols"], "sample": ds_facts["sample"],
                    "steps": dataset_steps},
        "scrape": {"title": "משרות ישראל · LinkedIn (חי)", "stats": il_facts["stats"],
                   "sample_cols": il_facts["sample_cols"], "sample": il_facts["sample"],
                   "steps": scrape_steps},
    }


def main():
    ds = build_dataset()
    il = build_scrape()
    data = content(ds, il)
    path = os.path.join(ASSETS, "explain.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("explain.json ->", path)
    print("DONE. Assets in frontend/assets/explain/")


if __name__ == "__main__":
    main()
