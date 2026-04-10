import argparse
import json
import re
import hashlib
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def pick_text(row):
    body = str(row.get("body")) if "body" in row and pd.notna(row.get("body")) else ""
    text = str(row.get("text")) if "text" in row and pd.notna(row.get("text")) else ""
    body = body.strip()
    text = text.strip()
    return body if body else text


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


def main():
    ap = argparse.ArgumentParser(
        description="Merge event labels from timelines into full candidate rows and emit per-user JSON."
    )
    ap.add_argument(
        "--candidates",
        type=Path,
        default=_REPO_ROOT / "output" / "candidates_v2.csv",
        help="Candidate posts CSV (same schema as filter_candidates_posts output).",
    )
    ap.add_argument(
        "--timelines",
        type=Path,
        default=_REPO_ROOT / "output" / "user_timelines.csv",
        help="Labeled timeline rows (row_key + event_label + author).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "output" / "user_activity.json",
        help="Output JSON: author -> list of {created_utc, text, event_label}.",
    )
    args = ap.parse_args()

    cands = pd.read_csv(args.candidates)
    timelines = pd.read_csv(args.timelines)

    if "author" not in cands.columns:
        cands["author"] = ""
    if "created_utc" not in cands.columns:
        cands["created_utc"] = ""

    cands["__text__"] = cands.apply(pick_text, axis=1)
    cands = cands[cands["__text__"].astype(str).str.strip().ne("")].copy()
    cands["author"] = cands["author"].astype(str)

    cands["row_key"] = cands.apply(
        lambda r: make_row_key(r.get("author"), r.get("created_utc"), r.get("__text__")),
        axis=1
    )

    timelines["row_key"] = timelines["row_key"].astype(str)
    timelines["event_label"] = timelines["event_label"].astype(str)

    label_map = dict(zip(timelines["row_key"], timelines["event_label"]))

    cands["event_label"] = cands["row_key"].map(label_map).fillna("None")

    authors = timelines["author"].dropna().astype(str).unique().tolist()
    cands_sub = cands[cands["author"].isin(authors)].copy()

    cands_sub["__created_dt"] = pd.to_datetime(cands_sub["created_utc"], errors="coerce", utc=True)
    mask_num = cands_sub["__created_dt"].isna() & cands_sub["created_utc"].astype(str).str.fullmatch(r"\d+")
    if mask_num.any():
        cands_sub.loc[mask_num, "__created_dt"] = pd.to_datetime(
            cands_sub.loc[mask_num, "created_utc"].astype(float), unit="s", utc=True
        )

    cands_sub = cands_sub.sort_values(by=["author", "__created_dt", "row_key"], kind="mergesort")

    user_activities = {}
    for author, g in cands_sub.groupby("author", sort=False):
        acts = []
        for _, r in g.iterrows():
            acts.append({
                "created_utc": str(r.get("created_utc", "")),
                "text": str(r.get("__text__", "")),
                "event_label": str(r.get("event_label", "None")),
            })
        user_activities[author] = acts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(user_activities, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out} with {len(user_activities)} users")


if __name__ == "__main__":
    main()
