# Modeling Career Trajectories in Computing (UofT EngSci Thesis)

This repository contains code and outputs for my undergraduate thesis on modeling longitudinal computing career trajectories from Reddit timelines using NLP, LLM-based event extraction, and association analyses (with planned causal extensions).

Supervisor: Prof. Michael Guerzhoy (University of Toronto)

---

## Project Overview

**Goal:** Construct user-level longitudinal timelines from Reddit and study how motivations and job-search behaviors are associated with downstream outcomes (e.g., receiving a job offer).

**Core pipeline:**
1. Collect Reddit comments at scale (10 years).
2. Build candidate sets for key events using keyword/regex filters.
3. Label candidate comments using an LLM (few-shot + strict rubric + temporal constraints).
4. Merge labeled events into per-user timelines, preserving non-event comments for context.
5. Detect **users motivations** using regex + LLM labeling.
6. Build **pre-anchor background features** using SentenceTransformer embeddings (then PCA).
7. Run **adjusted logistic regression** for association analyses.

---

## Data Sources

### Reddit data (10 years)
I downloaded historical Reddit data using the Arctic Shift Reddit Data Download Tool:
- https://arctic-shift.photon-reddit.com/download-tool

The working dataset is scoped around career discussions (e.g., `r/cscareerquestions`) and includes:
- author identifiers
- timestamps (`created_utc`)
- comment text (`body` / `text`)
