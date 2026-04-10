# Modeling Career Trajectories in Computing  
### UofT Engineering Science Thesis

This repository contains code, intermediate artifacts, and analysis outputs for my undergraduate thesis on modeling longitudinal computing career trajectories from Reddit using NLP, LLM-based event extraction, motivation classification, and survival analysis.

**Supervisor:** Prof. Michael Guerzhoy (University of Toronto)

---

## Project Overview

The goal of this project is to construct user-level longitudinal career timelines from Reddit discussions and study whether self-expressed motivations for pursuing computing are associated with how long it takes users to reach an interview-like outcome.

The thesis focuses on large-scale, time-stamped Reddit narratives as a complementary data source to traditional surveys or interviews. Using LLM-based labeling pipelines, unstructured comments are converted into structured event and motivation signals, which are then integrated into per-user trajectories for downstream time-to-event analysis.

### Research Question

**How does a user’s self-expressed motivation for pursuing a computing career relate to how long it takes them to reach their first interview-like outcome?**

---

## Core Pipeline

1. Collect historical Reddit comments at scale from career-related communities.
2. Preprocess the data by removing empty comments, normalizing timestamps, and deduplicating records.
3. Build candidate sets for career-related milestones using keyword and regex-based filtering.
4. Label candidate comments for career events using an LLM with few-shot prompting and a strict annotation rubric.
5. Label comments for self-expressed motivation using a separate LLM-based classification pipeline.
6. Construct chronological user-level timelines that preserve both labeled events and broader comment context.
7. Assign each user a **primary motivation** based on the most frequent non-`NONE` motivation label.
8. Run Kaplan–Meier survival analysis to study time from first observed expression of primary motivation to first interview-like outcome.

---

## Data Sources

### Reddit data

Historical Reddit comments were collected using the Arctic Shift Reddit Data Download Tool:

- https://arctic-shift.photon-reddit.com/download-tool

The working dataset is centered on career-related discussions, primarily from subreddits such as `r/cscareerquestions`, and includes:

- anonymized author identifiers
- timestamps (`created_utc`)
- comment text (`body` / `text`)

The dataset spans approximately 10 years of user activity and is used to build longitudinal user histories.

---

## Event Labeling

The event-labeling pipeline identifies career-related milestones from unstructured Reddit comments.

### Event labels

- `GRADUATION`
- `INTERVIEW`
- `GOT_AN_OFFER`
- `NONE`

### Event-labeling approach

- Use rule-based candidate filtering to identify comments likely to mention career milestones.
- Apply LLM-based classification to the candidate set using few-shot prompting.
- Use a strict rubric requiring that:
  - the event refers to the **author** of the comment
  - the event is temporally grounded near the comment date
  - ambiguous, hypothetical, second-hand, or vague comments are labeled `NONE`

This stage extracts discrete, time-stamped events that can be placed into user trajectories.

---

## Motivation Labeling

A separate LLM-based pipeline is used to identify the author’s self-expressed motivation for pursuing a computing career or educational pathway.

### Motivation labels

- `MONEY`
- `INTEREST`
- `CAREER_SWITCH`
- `STABILITY`
- `PRESTIGE`
- `NONE`

### Motivation definitions

- `MONEY`: financial goals, salary, compensation, or financial security
- `INTEREST`: intrinsic enjoyment of programming, computers, or technology
- `CAREER_SWITCH`: transition into computing from another field
- `STABILITY`: desire for job security or long-term career prospects
- `PRESTIGE`: status, brand-name companies, or social prestige
- `NONE`: no clear personal motivation expressed

After comment-level classification, each user is assigned a **primary motivation** based on their most frequent non-`NONE` label.

---

## User-Level Timeline Construction

For each user with labeled career events, all available historical comments are ordered chronologically to build a longitudinal timeline.

Each user trajectory may contain:

- labeled career events
- motivation-labeled comments
- non-event comments preserved as narrative context

This allows the project to model career development as an ordered sequence rather than a single static outcome.

---

## Analysis Design

The main analysis uses **survival analysis** to compare time-to-event patterns across motivation groups.

### Start time

- first observed expression of the user’s **primary motivation**

### Event of interest

- first **interview-like outcome**
- operationalized as the first occurrence of:
  - `INTERVIEW`, or
  - `GOT_AN_OFFER`

### Method

- Kaplan–Meier survival curves
- global log-rank test

### Observation window

- 1460 days

This framework captures not only whether users appear to reach an interview-like milestone, but also how quickly they do so.

---

## Main Thesis Finding

Across the 1460-day observation window, survival distributions differed significantly across motivation groups.

Among the larger groups, users in the `INTEREST` category generally reached interview-like outcomes earlier than users in the `MONEY` and `CAREER_SWITCH` categories. The smaller `STABILITY` and `PRESTIGE` groups showed the highest eventual cumulative proportions reaching an interview-like event, though those estimates should be interpreted more cautiously due to smaller sample sizes.

These findings suggest that self-expressed motivation is associated with differences in job-search progression in computing pathways.

---

## Ethical Notes

This project uses publicly available Reddit data, but treats it with care consistent with prior work on ethical Reddit research.

- user identifiers are anonymized
- analyses are conducted at the aggregate level
- the focus is on statistical patterns rather than individual case studies
- direct quotation of user content should be avoided where possible

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`scripts/filter_candidates_posts.py`](scripts/filter_candidates_posts.py) | Regex-based candidate extraction from Reddit JSONL |
| [`scripts/make_label_set.py`](scripts/make_label_set.py) | Stratified sample for manual / few-shot seed labels |
| [`scripts/label_and_build_timelines.py`](scripts/label_and_build_timelines.py) | LLM event labeling and `user_timelines.csv` construction |
| [`scripts/build_user_activity.py`](scripts/build_user_activity.py) | Merge labels into per-user `user_activity.json` |
| [`scripts/user_motivation.py`](scripts/user_motivation.py) | LLM motivation classification over user comments |
| [`scripts/motivation_analysis.py`](scripts/motivation_analysis.py) | Survival analysis (Kaplan–Meier, log-rank, Cox) |

Intermediate files are written under `output/` (ignored in git). Clone the repo, create `output/`, and run scripts from the **repository root** so default paths resolve correctly.

### Quick start (after installing dependencies)

```bash
python -m venv .venv
# activate .venv, then:
pip install -r requirements.txt
export OPENAI_API_KEY=...   # Windows: set OPENAI_API_KEY=...

python scripts/filter_candidates_posts.py --input /path/to/posts.jsonl
python scripts/make_label_set.py --total 500
# Manually fill event_type (etc.) in output/label_500.csv, then:
python scripts/label_and_build_timelines.py
python scripts/build_user_activity.py
python scripts/user_motivation.py --input output/user_activity.json --outdir output/motivation_outputs
python scripts/motivation_analysis.py
```

---

## Repository Scope

This repository includes code and outputs for:

- Reddit preprocessing
- candidate filtering
- LLM-based event labeling
- LLM-based motivation labeling
- user-level timeline construction
- survival analysis and visualization

Some earlier experimental directions may still appear in the codebase, but the final thesis centers on the event/motivation timeline pipeline and survival-based analysis described above.

---

## Future Directions

Possible extensions include:

- modeling motivation as dynamic rather than fixed at the user level
- incorporating richer background features from prior text
- testing robustness under alternative start-time or event definitions
- exploring stronger adjustment or causal inference frameworks in future work

---