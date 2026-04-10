import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "output"
_MOT = _OUT / "motivation_outputs"

# ---------------- Paths ----------------
USER_ACTIVITY = _OUT / "user_activity.json"
COMMENT_MOT = _MOT / "comment_motivations.jsonl"
USER_MOT = _MOT / "user_motivations.csv"

# ---------------- Settings ----------------
HORIZON_DAYS = 1460  
DROP_NONE = True

INTERVIEW_EVENT = "Interview"
OFFER_EVENT = "Got an Offer"

MOTIVATION_ORDER = ["MONEY", "CAREER_SWITCH", "INTEREST", "STABILITY", "PRESTIGE"]

# ---------------- Load ----------------
with open(USER_ACTIVITY, "r", encoding="utf-8") as f:
    activity = json.load(f)

comment_mot = pd.read_json(COMMENT_MOT, lines=True)
comment_mot["created_utc"] = pd.to_datetime(comment_mot["created_utc"], errors="coerce")

user_mot = pd.read_csv(USER_MOT)

comment_mot_by_user = dict(tuple(comment_mot.groupby("user")))
top_mot_by_user = dict(zip(user_mot["user"], user_mot["top_motivation"]))

# ---------------- Build user-level times ----------------
rows = []
for user, comments in activity.items():
    df = pd.DataFrame(comments)
    if df.empty:
        continue

    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
    df = df.sort_values("created_utc").copy()

    top_mot = top_mot_by_user.get(user, "NONE")

    # comment-level motivation rows for this user
    if user in comment_mot_by_user:
        cmu = comment_mot_by_user[user].copy()
        cmu["created_utc"] = pd.to_datetime(cmu["created_utc"], errors="coerce")
        cmu = cmu.sort_values("created_utc")
    else:
        cmu = pd.DataFrame(columns=["created_utc", "label"])

    # A) start definition 1: first expression of user's dominant motivation
    if top_mot != "NONE" and not cmu.empty:
        dominant_start_candidates = cmu.loc[cmu["label"] == top_mot, "created_utc"]
        start_time_dominant = (
            dominant_start_candidates.min()
            if not dominant_start_candidates.empty
            else pd.NaT
        )
    else:
        start_time_dominant = pd.NaT

    # B) start definition 2: first motivation-labeled comment of any kind
    first_motivation_time = cmu["created_utc"].min() if not cmu.empty else pd.NaT

    # -------- Event definitions --------
    # interview-only
    interview_candidates = df.loc[df["event_label"] == INTERVIEW_EVENT, "created_utc"]
    first_interview_only = (
        interview_candidates.min() if not interview_candidates.empty else pd.NaT
    )

    # offer-only
    offer_candidates = df.loc[df["event_label"] == OFFER_EVENT, "created_utc"]
    first_offer = offer_candidates.min() if not offer_candidates.empty else pd.NaT

    # interview-like = interview OR offer
    interview_like_candidates = df.loc[
        df["event_label"].isin([INTERVIEW_EVENT, OFFER_EVENT]), "created_utc"
    ]
    first_interview_like = (
        interview_like_candidates.min() if not interview_like_candidates.empty else pd.NaT
    )

    rows.append(
        {
            "user": user,
            "top_motivation": top_mot,
            "start_time_dominant": start_time_dominant,
            "start_time_first_motivation": first_motivation_time,
            "first_interview_only": first_interview_only,
            "first_offer": first_offer,
            "first_interview_like": first_interview_like,
        }
    )

timeline = pd.DataFrame(rows)

if DROP_NONE:
    timeline = timeline[timeline["top_motivation"].fillna("NONE") != "NONE"].copy()

# ---------------- Helper: build censored durations ----------------
def make_survival_df(df, start_col, event_col, label):
    """
    Creates:
      {label}_duration_days  : observed duration if event happens within horizon else HORIZON_DAYS
      {label}_event_observed : 1 if event observed within horizon else 0
    """
    tmp = df.dropna(subset=[start_col]).copy()

    true_dur = (tmp[event_col] - tmp[start_col]).dt.total_seconds() / (3600 * 24)
    event_obs = true_dur.notna() & (true_dur >= 0) & (true_dur <= HORIZON_DAYS)

    tmp[f"{label}_duration_days"] = true_dur.where(event_obs, HORIZON_DAYS)
    tmp[f"{label}_event_observed"] = event_obs.astype(int)
    return tmp

# ---------------- Reporting helpers ----------------
def print_sizes(df, duration_col, event_col, title):
    print(f"\n{title}")
    g = (
        df.groupby("top_motivation")
        .agg(
            N=("user", "size"),
            events=(event_col, "sum"),
            censored=(event_col, lambda x: (1 - x).sum()),
            median_duration=(duration_col, "median"),
        )
        .sort_values("N", ascending=False)
    )
    print(g)

def plot_km(df, duration_col, event_col, title):
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for mot in [m for m in MOTIVATION_ORDER if m in df["top_motivation"].unique()]:
        sub = df[df["top_motivation"] == mot]
        if sub.empty:
            continue
        kmf.fit(
            sub[duration_col],
            event_observed=sub[event_col],
            label=f"{mot} (N={len(sub)})",
        )
        kmf.plot_survival_function()

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Survival probability (not yet reached event)")
    plt.tight_layout()
    plt.show()

def km_logrank_tests(df, duration_col, event_col, group_col="top_motivation", do_pairwise=True):
    global_res = multivariate_logrank_test(
        event_durations=df[duration_col],
        groups=df[group_col],
        event_observed=df[event_col],
    )

    print("\n========== Log-rank test (global) ==========")
    print(f"Outcome: {duration_col} / {event_col} grouped by {group_col}")
    print(f"Chi-square: {global_res.test_statistic:.4f}")
    print(f"p-value:    {global_res.p_value:.6g}")

    if do_pairwise:
        pw = pairwise_logrank_test(
            df[duration_col],
            df[group_col],
            df[event_col],
        )
        print("\n========== Log-rank tests (pairwise p-values) ==========")
        print(pw.p_value)

    return global_res

def fit_cox(df, duration_col, event_col, title):
    cox_df = df[["top_motivation", duration_col, event_col]].copy()
    cox_df = pd.get_dummies(cox_df, columns=["top_motivation"], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)

    print(f"\nCox PH model: {title}")
    print(cph.summary)

def run_survival_analysis(df, start_col, event_col, label, title_prefix):
    surv = make_survival_df(df, start_col, event_col, label)

    duration_col = f"{label}_duration_days"
    observed_col = f"{label}_event_observed"

    print_sizes(
        surv, duration_col, observed_col,
        f"{title_prefix} (with censoring at {HORIZON_DAYS} days)"
    )

    plot_km(
        surv, duration_col, observed_col,
        f"Kaplan–Meier: {title_prefix} (censor at {HORIZON_DAYS} days)"
    )

    km_logrank_tests(surv, duration_col, observed_col, do_pairwise=True)
    fit_cox(surv, duration_col, observed_col, title_prefix)

    return surv

# ---------------- Analyses to run ----------------
ANALYSES = [
    # Current main analysis, renamed explicitly
    {
        "start_col": "start_time_dominant",
        "event_col": "first_interview_like",
        "label": "dom_to_interviewlike",
        "title": "Dominant-motivation start → Interview-or-Offer",
    },

    # Event-definition sensitivity
    {
        "start_col": "start_time_dominant",
        "event_col": "first_interview_only",
        "label": "dom_to_interviewonly",
        "title": "Dominant-motivation start → Interview only",
    },
    {
        "start_col": "start_time_dominant",
        "event_col": "first_offer",
        "label": "dom_to_offeronly",
        "title": "Dominant-motivation start → Offer only",
    },

    # Start-time sensitivity
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_interview_like",
        "label": "firstmot_to_interviewlike",
        "title": "First-motivation-comment start → Interview-or-Offer",
    },
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_interview_only",
        "label": "firstmot_to_interviewonly",
        "title": "First-motivation-comment start → Interview only",
    },
    {
        "start_col": "start_time_first_motivation",
        "event_col": "first_offer",
        "label": "firstmot_to_offeronly",
        "title": "First-motivation-comment start → Offer only",
    },
]

all_surv_results = {}

for spec in ANALYSES:
    print("\n" + "=" * 90)
    print(f"RUNNING: {spec['title']}")
    print("=" * 90)

    surv_df = run_survival_analysis(
        timeline,
        start_col=spec["start_col"],
        event_col=spec["event_col"],
        label=spec["label"],
        title_prefix=spec["title"],
    )
    all_surv_results[spec["label"]] = surv_df


# ============================================================
# 6. Year / cohort stratification
# ============================================================

# Choose which start definition defines "cohort"
# If you already added start_time_first_motivation in your updated script, use that.
# Otherwise replace with "start_time".
COHORT_START_COL = "start_time_dominant"   # or "start_time"

# Main survival outcome to stratify
# Recommended: interview-or-offer because it was your more stable main result
COHORT_EVENT_COL = "first_interview_like"          # or "first_interview" / "first_offer"
COHORT_LABEL = "cohort_interviewlike"
COHORT_TITLE = "Cohort-stratified: Start → Interview-or-Offer"


def add_cohort_columns(df, start_col):
    tmp = df.copy()
    tmp = tmp.dropna(subset=[start_col]).copy()

    tmp["start_year"] = pd.to_datetime(tmp[start_col], errors="coerce").dt.year

    # ---- Option A: fixed bins (easy to interpret) ----
    def map_cohort_fixed(y):
        if pd.isna(y):
            return pd.NA
        y = int(y)
        if y <= 2019:
            return "<=2019"
        elif y <= 2021:
            return "2020-2021"
        else:
            return "2022+"

    tmp["cohort_fixed"] = tmp["start_year"].apply(map_cohort_fixed)

    # ---- Option B: data-driven terciles (better balanced Ns) ----
    # Keeps cohorts more even if your years are imbalanced
    valid_years = tmp["start_year"].dropna()
    if valid_years.nunique() >= 3:
        q1 = valid_years.quantile(1/3)
        q2 = valid_years.quantile(2/3)

        def map_cohort_tercile(y):
            if pd.isna(y):
                return pd.NA
            if y <= q1:
                return "Early"
            elif y <= q2:
                return "Middle"
            else:
                return "Late"

        tmp["cohort_tercile"] = tmp["start_year"].apply(map_cohort_tercile)
    else:
        tmp["cohort_tercile"] = "All"

    return tmp


def print_cohort_counts(df, cohort_col):
    print(f"\n========== Cohort counts: {cohort_col} ==========")
    print(df[cohort_col].value_counts(dropna=False).sort_index())

    print(f"\n========== Cohort x Motivation counts: {cohort_col} ==========")
    print(pd.crosstab(df[cohort_col], df["top_motivation"]))


def run_within_cohort_survival(
    df,
    cohort_col,
    start_col,
    event_col,
    label_prefix,
    min_total_n=40,
    min_group_n=10
):
    """
    Runs the same survival analysis separately within each cohort.
    Skips cohorts that are too small overall or have very tiny motivation groups.
    """
    cohorts = [c for c in df[cohort_col].dropna().unique()]

    for cohort in sorted(cohorts):
        sub = df[df[cohort_col] == cohort].copy()

        # overall size check
        if len(sub) < min_total_n:
            print(f"\n[SKIP] cohort={cohort}: total N={len(sub)} < {min_total_n}")
            continue

        # keep only motivation groups with enough users
        keep_mots = sub["top_motivation"].value_counts()
        keep_mots = keep_mots[keep_mots >= min_group_n].index.tolist()
        sub = sub[sub["top_motivation"].isin(keep_mots)].copy()

        if sub["top_motivation"].nunique() < 2:
            print(f"\n[SKIP] cohort={cohort}: fewer than 2 motivation groups with N>={min_group_n}")
            continue

        surv = make_survival_df(sub, start_col, event_col, f"{label_prefix}_{cohort}")

        duration_col = f"{label_prefix}_{cohort}_duration_days"
        event_obs_col = f"{label_prefix}_{cohort}_event_observed"

        print("\n" + "=" * 90)
        print(f"COHORT: {cohort}")
        print("=" * 90)

        print_sizes(
            surv,
            duration_col,
            event_obs_col,
            f"{COHORT_TITLE} | cohort={cohort}"
        )

        plot_km(
            surv,
            duration_col,
            event_obs_col,
            f"{COHORT_TITLE} | cohort={cohort} (censor at {HORIZON_DAYS} days)"
        )

        km_logrank_tests(
            surv,
            duration_col,
            event_obs_col,
            do_pairwise=True
        )

        # optional Cox only if enough events
        if surv[event_obs_col].sum() >= 20:
            fit_cox(
                surv,
                duration_col,
                event_obs_col,
                f"{COHORT_TITLE} | cohort={cohort}"
            )
        else:
            print(f"[SKIP COX] cohort={cohort}: too few observed events")


# ---------------- Build cohort variables ----------------
timeline_cohort = add_cohort_columns(timeline, COHORT_START_COL)

# Check both versions
print_cohort_counts(timeline_cohort, "cohort_fixed")
print_cohort_counts(timeline_cohort, "cohort_tercile")

# ---------------- Run stratified analyses ----------------
# 1) fixed calendar bins
run_within_cohort_survival(
    timeline_cohort,
    cohort_col="cohort_fixed",
    start_col=COHORT_START_COL,
    event_col=COHORT_EVENT_COL,
    label_prefix="fixedcohort"
)

# 2) balanced tercile bins
run_within_cohort_survival(
    timeline_cohort,
    cohort_col="cohort_tercile",
    start_col=COHORT_START_COL,
    event_col=COHORT_EVENT_COL,
    label_prefix="tercilecohort"
)

# ============================================================
# Final percentage: % of each motivation that got interview within 2 years
# ============================================================

def print_event_percentages(df, event_col, title):
    summary = (
        df.groupby("top_motivation")
        .agg(
            total_users=("user", "size"),
            users_with_event=(event_col, "sum"),
        )
        .reset_index()
    )

    summary["percent_with_event"] = (
        100 * summary["users_with_event"] / summary["total_users"]
    )

    summary = summary.sort_values("percent_with_event", ascending=False)

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(summary.to_string(index=False))

    return summary


# Example 1: dominant motivation start -> interview OR offer within 2 years
main_surv = all_surv_results["dom_to_interviewlike"]

main_percentage_summary = print_event_percentages(
    main_surv,
    event_col="dom_to_interviewlike_event_observed",
    title="Percentage of users in each motivation group who got an interview-or-offer within 2 years"
)

print("\nFinal percentages by motivation:")
for _, row in main_percentage_summary.iterrows():
    print(
        f"{row['top_motivation']}: "
        f"{row['percent_with_event']:.1f}% "
        f"({int(row['users_with_event'])}/{int(row['total_users'])})"
    )