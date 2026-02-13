import os
import json
import re
import hashlib
import pandas as pd

# ----------------------------
# PATHS (edit as needed)
# ----------------------------
CANDS_PATH = r"C:\Users\lucia\Documents\thesis\output\candidates_v2.csv"
TL_PATH    = r"C:\Users\lucia\Documents\thesis\output\user_timelines.csv"
OUT_PATH   = r"C:\Users\lucia\Documents\thesis\output\user_activity.json"

# ----------------------------
# TEXT HELPER
# ----------------------------
def pick_text(row):
    body = str(row.get("body")) if "body" in row and pd.notna(row.get("body")) else ""
    text = str(row.get("text")) if "text" in row and pd.notna(row.get("text")) else ""
    body = body.strip()
    text = text.strip()
    return body if body else text

# ----------------------------
# ROW_KEY (must match server script)
# ----------------------------
_WS_RE = re.compile(r"\s+")

def _norm_text(x):
    x = "" if x is None else str(x)
    return _WS_RE.sub(" ", x.strip())

def _to_iso_for_key(v):
    s = "" if v is None else str(v).strip()
    if not s:
        return ""
    if s.isdigit():
        try:
            dt = pd.to_datetime(float(s), unit="s", utc=True)
            return dt.strftime("%Y-%m-%d %H:%M:%S%z")
        except Exception:
            pass
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(dt):
        return s
    return dt.strftime("%Y-%m-%d %H:%M:%S%z")

def make_row_key(author, created_utc, body):
    sig = f"{author or ''}|{_to_iso_for_key(created_utc)}|{_norm_text(body)}"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()

# ----------------------------
# LOAD
# ----------------------------
cands = pd.read_csv(CANDS_PATH)
timelines = pd.read_csv(TL_PATH)

# ensure columns exist
if "author" not in cands.columns:
    cands["author"] = ""
if "created_utc" not in cands.columns:
    cands["created_utc"] = ""

# text + key for candidates
cands["__text__"] = cands.apply(pick_text, axis=1)
cands = cands[cands["__text__"].astype(str).str.strip().ne("")].copy()
cands["author"] = cands["author"].astype(str)

cands["row_key"] = cands.apply(
    lambda r: make_row_key(r.get("author"), r.get("created_utc"), r.get("__text__")),
    axis=1
)

# label map from user_timelines.csv (event rows only)
# IMPORTANT: this assumes TL file has columns row_key + event_label (your server script does)
timelines["row_key"] = timelines["row_key"].astype(str)
timelines["event_label"] = timelines["event_label"].astype(str)

label_map = dict(zip(timelines["row_key"], timelines["event_label"]))

# attach labels to *all* comments
cands["event_label"] = cands["row_key"].map(label_map).fillna("None")

# restrict to authors we care about (authors with >=1 labeled event)
authors = timelines["author"].dropna().astype(str).unique().tolist()
cands_sub = cands[cands["author"].isin(authors)].copy()

# sort by time (parse for sorting only)
cands_sub["__created_dt"] = pd.to_datetime(cands_sub["created_utc"], errors="coerce", utc=True)
mask_num = cands_sub["__created_dt"].isna() & cands_sub["created_utc"].astype(str).str.fullmatch(r"\d+")
if mask_num.any():
    cands_sub.loc[mask_num, "__created_dt"] = pd.to_datetime(
        cands_sub.loc[mask_num, "created_utc"].astype(float), unit="s", utc=True
    )

cands_sub = cands_sub.sort_values(by=["author", "__created_dt", "row_key"], kind="mergesort")

# ----------------------------
# BUILD JSON
# ----------------------------
user_activities = {}
for author, g in cands_sub.groupby("author", sort=False):
    acts = []
    for _, r in g.iterrows():
        acts.append({
            "created_utc": str(r.get("created_utc", "")),
            "text": str(r.get("__text__", "")),
            "event_label": str(r.get("event_label", "None")),
            # optional extras if available in your file:
            # "subreddit": r.get("subreddit", ""),
            # "id": r.get("id", ""),
            # "row_key": r.get("row_key", ""),
        })
    user_activities[author] = acts

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(user_activities, f, ensure_ascii=False, indent=2)

print(f"wrote {OUT_PATH} with {len(user_activities)} users")
