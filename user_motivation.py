#!/usr/bin/env python3
import os, json, hashlib, argparse, time
from typing import Dict, Any, Set
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0

LABELS = ["MONEY", "INTEREST", "CAREER_SWITCH", "STABILITY", "PRESTIGE", "NONE"]
LABEL_DEFS = {
    "MONEY": "Seeking higher salary, total compensation, or financial security.",
    "INTEREST": "Enjoyment of programming, computers, or intrinsic interest in tech.",
    "CAREER_SWITCH": "Switching or pivoting into CS/SWE from another field/role.",
    "STABILITY": "Job security, reliable long-term prospects.",
    "PRESTIGE": "Status/brand: big-name companies, top-tier prestige.",
    "NONE": "No clear personal motivation expressed (or not about author's motive).",
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def make_comment_id(user: str, created_utc: str, text: str) -> str:
    base = f"{user}|{created_utc}|{text}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def build_prompt(text: str) -> str:
    defs = "\n".join([f"- {k}: {v}" for k, v in LABEL_DEFS.items()])
    return f"""
Classify the author's PERSONAL motivation for pursuing CS career/education from the single comment below.

Choose EXACTLY ONE label from: {", ".join(LABELS)}

Definitions:
{defs}

Rules:
- Use NONE unless the author's OWN motivation is clearly supported by the text.
- Advice/opinions/general observations not about the author → NONE.
- evidence must be a short direct quote from the comment (or "" if NONE).
- confidence must be 0 to 1.
- Output JSON only.

JSON schema:
{{
  "label": "MONEY|INTEREST|CAREER_SWITCH|STABILITY|PRESTIGE|NONE",
  "confidence": 0.0,
  "evidence": "short quote or empty string"
}}

Comment:
\"\"\"{text.strip()}\"\"\"
""".strip()

def classify_comment(text: str, max_retries: int = 5) -> Dict[str, Any]:
    prompt = build_prompt(text)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a careful and conservative classifier."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                timeout=60,
            )
            out = json.loads(resp.choices[0].message.content)

            # Validate defensively
            if out.get("label") not in LABELS:
                return {"label": "NONE", "confidence": 0.0, "evidence": ""}

            conf = float(out.get("confidence", 0.0))
            if not (0.0 <= conf <= 1.0):
                conf = 0.0

            ev = out.get("evidence", "")
            if not isinstance(ev, str):
                ev = ""

            return {"label": out["label"], "confidence": conf, "evidence": ev}

        except Exception as e:
            last_err = str(e)
            # exponential backoff
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)

    # Final fallback
    return {"label": "NONE", "confidence": 0.0, "evidence": "" , "error": last_err}

def load_done_ids(jsonl_path: str) -> Set[str]:
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("comment_id")
                if cid:
                    done.add(cid)
            except Exception:
                continue
    return done

def aggregate_user_csv(jsonl_path: str, out_csv: str) -> None:
    user_mass = defaultdict(lambda: defaultdict(float))
    user_total = defaultdict(float)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user = obj["user"]
            label = obj["label"]
            conf = float(obj.get("confidence", 0.0))
            if label != "NONE":
                user_mass[user][label] += conf
                user_total[user] += conf

    rows = []
    for user in user_total.keys() | user_mass.keys():
        total = user_total.get(user, 0.0)
        props = {lab: (user_mass[user].get(lab, 0.0) / total if total > 0 else 0.0)
                 for lab in LABELS if lab != "NONE"}
        top = max(props, key=props.get) if total > 0 else "NONE"
        rows.append({
            "user": user,
            "top_motivation": top,
            "top_motivation_prop": props.get(top, 0.0),
            "total_confidence_mass": total,
            **{f"prop_{lab}": props.get(lab, 0.0) for lab in props},
            **{f"mass_{lab}": user_mass[user].get(lab, 0.0) for lab in props},
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to user_activity.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--resume", action="store_true", help="Resume from existing JSONL")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_jsonl = os.path.join(args.outdir, "comment_motivations.jsonl")
    out_csv = os.path.join(args.outdir, "user_motivations.csv")

    done_ids = load_done_ids(out_jsonl) if args.resume else set()
    if args.resume:
        print(f"[resume] Loaded {len(done_ids):,} already-labeled comments.")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Append if resume, else overwrite
    mode = "a" if args.resume else "w"
    with open(out_jsonl, mode, encoding="utf-8") as fout:
        for user, comments in tqdm(data.items(), desc="Users"):
            if not isinstance(comments, list):
                continue
            for c in comments:
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                created = str(c.get("created_utc") or "")

                cid = make_comment_id(user, created, text)
                if cid in done_ids:
                    continue

                pred = classify_comment(text)

                rec = {
                    "comment_id": cid,
                    "user": user,
                    "created_utc": created,
                    "event_label": c.get("event_label"),
                    "label": pred["label"],
                    "confidence": pred["confidence"],
                    "evidence": pred["evidence"],
                }
                if "error" in pred:
                    rec["error"] = pred["error"]

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()  # crucial for crash safety
                done_ids.add(cid)

    aggregate_user_csv(out_jsonl, out_csv)
    print(f"Done.\n- {out_jsonl}\n- {out_csv}")

if __name__ == "__main__":
    main()
