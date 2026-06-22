# Redrob Intelligent Candidate Ranking System

> An AI-powered candidate ranking system that understands *context*, not keywords.

Built for the **Redrob Intelligent Candidate Discovery & Ranking Challenge** (100,000 candidates → top 100 for a Senior AI Engineer role).

---

## 🎯 The Problem

Recruiters filter candidates using keyword matching. This fails in three ways:

1. **The Tier-5 trap** — A senior engineer who built production recommendation systems may never write the words "RAG" or "Pinecone" anywhere in their profile.
2. **The keyword-stuffer trap** — A Marketing Manager can have a skills section full of "Python, TensorFlow, PyTorch, LangChain" but has never built a model.
3. **The inactive-candidate trap** — A perfect-on-paper candidate who hasn't logged in for six months is, for hiring purposes, not actually available.

The challenge JD says it explicitly:

> *"The right answer is not finding candidates whose skills section contains AI keywords. A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."*

---

## 💡 Our Solution

A **two-phase hybrid AI ranker** that combines semantic understanding, lexical precision, behavioral signals, and structured feature engineering.

### Pipeline Overview

```
Phase 1 — Offline (one-time, ~10 minutes)
┌──────────────────────────────────────────────────────────┐
│  100K candidates                                          │
│       │                                                   │
│       ├──► BGE-small-en embeddings → FAISS index         │
│       └──► BM25 lexical index                             │
└──────────────────────────────────────────────────────────┘

Phase 2 — Online (rank, ~2 minutes, 5-min budget)
┌──────────────────────────────────────────────────────────┐
│  Job Description                                          │
│       │                                                   │
│       ├──► Dense retrieval (top 2500)                     │
│       ├──► BM25 retrieval (top 2500)                      │
│       ├──► RRF fusion → top 3000                          │
│       │                                                   │
│       ├──► Feature-based re-ranking (9 signals)          │
│       ├──► Honeypot filter                                │
│       ├──► Active-candidate gate                          │
│       └──► Top 100 + reasoning → submission.csv          │
└──────────────────────────────────────────────────────────┘
```

---

## 🧮 Scoring Formula

```python
final_score = (
    0.28 * retrieval_experience    # JD core: retrieval/ranking/recommendation
  + 0.18 * semantic_similarity     # BGE cosine similarity
  + 0.15 * product_company_score    # Penalizes service-only careers
  + 0.10 * evaluation_experience   # NDCG, A/B testing, experimentation
  + 0.08 * behavioral_signals       # Response rate, activity, open-to-work
  + 0.07 * experience_band          # 5-9 years ideal (JD-specified band)
  + 0.06 * tool_match              # FAISS, Pinecone, Weaviate, etc.
  + 0.04 * career_progression       # Upward trajectory, not title-chasing
  + 0.04 * education_tier           # Institution prestige
  + 0.02 * github_activity          # Production evidence
)
final_score *= honeypot_trust        # Multiplicative honeypot penalty
final_score *= active_availability   # Multiplicative availability penalty
```

### What Each Signal Catches

| Signal | Catches the trap of |
|--------|---------------------|
| **retrieval_experience** | Tier-5 candidates who describe their work in plain language ("built search system") rather than naming tools |
| **semantic_similarity** | Synonyms and adjacent concepts ("relevance" ≈ "matching" ≈ "ranking") |
| **product_company_score** | Career-long service-company employees (TCS, Infosys, Wipro) |
| **evaluation_experience** | Candidates who've actually shipped ranking systems (vs. only built prototypes) |
| **behavioral_signals** | Inactive candidates who won't actually engage with recruiters |
| **experience_band** | Over-qualified staff engineers or under-qualified mid-levels |
| **tool_match** | Specific tools named in the JD: FAISS, Pinecone, Weaviate, Elasticsearch, sentence-transformers |
| **career_progression** | Title-chasers who jump every 1.5 years for the next level |
| **education_tier** | Top-tier institutions (small bonus; not a hard filter) |
| **github_activity** | Engineers with public evidence of shipping code |

---

## 📁 Project Structure

```
redrob-ranker/
├── data/
│   ├── candidates.jsonl              # 100K candidate pool
│   ├── job_description.txt           # Target JD
│   └── candidate_schema.json         # Field reference
│
├── artifacts/                        # Pre-computed (Phase 1 output)
│   ├── faiss.index                   # Dense vector index
│   ├── candidate_embeddings.npy      # Raw embedding matrix
│   ├── candidate_ids.json            # ID ↔ index mapping
│   └── bm25_index.pkl                # Lexical index
│
├── output/
│   └── submission.csv                # Final top-100 ranking
│
├── scripts/
│   └── build_artifacts.py            # Phase 1 entry point
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Constants, weights, thresholds
│   ├── loader.py                     # JSONL / docx loaders
│   ├── embedding.py                  # Sentence-transformer wrappers
│   ├── bm25_index.py                 # BM25 build/search
│   ├── features.py                   # 9 engineered signals
│   ├── honeypot.py                   # Multi-pattern honeypot detector
│   ├── career.py                     # Career progression scoring
│   ├── active_gate.py                # Availability filter
│   ├── hybrid_retrieval.py           # BM25 + dense via RRF
│   ├── ranker.py                     # Final scoring + top-100 selection
│   ├── reasoning.py                  # Per-candidate explanations
│   └── generate_submission.py        # Phase 2 entry point
│
├── validate_submission.py            # Format validator (provided by challenge)
├── requirements.txt
├── run.bat                           # Windows entry point
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 16 GB RAM
- ~2 GB disk for model + embeddings

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Pipeline

**Windows:**

```powershell
python -m scripts.build_artifacts    # Phase 1: ~10 minutes
python -m src.generate_submission     # Phase 2: ~2 minutes
python validate_submission.py output\submission.csv
```

**Linux / macOS:**

```bash
python -m scripts.build_artifacts
python -m src.generate_submission
python validate_submission.py output/submission.csv
```

### Expected Output

```
Loading candidates...
  100000 candidates in 12.3s
Encoding with sentence-transformers...
  embeddings shape: (100000, 384) in 380.5s
Building BM25 index...
ALL DONE in 14420.1s

[1/5] Loading artifacts...
[2/5] Loading candidates.jsonl...
      Loaded 100000 candidates in 6.6s
[3/5] Encoding job description...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 10401.22it/s]
[4/5] Hybrid retrieval + scoring...
      Got 100 ranked candidates in 35.8s
[5/5] Writing submission...
DONE in 35.8s
Submission is valid.
```

---
## 📦 Note on Large Files

The following files are excluded from this repository due to GitHub's 100 MB file size limit:

- `data/candidates.jsonl` (~465 MB) — provided by the challenge organizers
- `artifacts/*.npy`, `artifacts/*.index`, `artifacts/*.pkl` (~480 MB) — regenerable artifacts

To regenerate the artifacts after cloning:

```bash
# Place candidates.jsonl in data/
python -m scripts.build_artifacts    # ~3 hours
python -m src.generate_submission     # ~2 minutes

---

## 🛠️ Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | `BAAI/bge-small-en-v1.5` | ~33 MB, 384-dim, strong retrieval quality, CPU-friendly |
| **Vector search** | FAISS (`IndexFlatIP`) | Exact cosine similarity, no GPU required |
| **Lexical search** | `rank_bm25` | Pure-Python BM25, recovers exact tool mentions |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Standard hybrid retrieval technique |
| **Data** | pandas, numpy | Standard tooling |
| **Validation** | jsonschema | Match challenge's candidate_schema.json |

---

## 🎯 Key Design Decisions

### 1. Hybrid Retrieval Beats Single-Method
- **Pure dense** misses candidates who mention specific tools (FAISS, Pinecone, NDCG) but don't describe their work in semantically rich ways.
- **Pure BM25** misses Tier-5 candidates who describe work in plain language.
- **RRF fusion** gets the best of both with no hyperparameter tuning.

### 2. Behavioral Signals Are First-Class
The JD explicitly down-weights inactive candidates. Our `active_gate.py` hard-filters candidates who:
- Haven't logged in for 9+ months
- Have <5% recruiter response rate AND aren't open-to-work

### 3. Honeypot Detection Is Multi-Signal
The dataset contains ~80 honeypots with subtly-impossible profiles. We detect them via:
- Years-of-experience vs career-duration mismatch (>4 years off)
- Title seniority vs years ("Principal Engineer" with 2 years)
- Skill proficiency vs declared duration ("Expert" with 0 months)
- Phantom expert skills (declared expert but never used in any role)
- "Too-perfect" signal patterns (multiple perfect-100% signals → likely synthetic)
- Excessive skill count for short career

### 4. CPU-Only Constraint
The challenge enforces 5 min, 16 GB, CPU only, no network during ranking. Every component is CPU-compatible:
- `bge-small-en` runs fast on CPU (~5 min for 100K candidates on a modern laptop)
- FAISS `IndexFlatIP` is exact (no GPU quantization needed for 100K vectors)
- BM25 via `rank_bm25` is pure Python

### 5. Career Progression Over Raw Tenure
The JD explicitly rejects title-chasers. Our `career.py`:
- Rewards upward title trajectories
- Penalizes short stints (<14 months, non-current)
- Penalizes level drops (signals instability)
- Rewards 2+ year average tenure

### 6. Evidence-Based Reasoning
Judges manually inspect the `reasoning` column. We don't generate generic templates; each row's explanation references the candidate's specific:
- Years of experience + current title
- Top strengths (retrieval/ranking/recommendation/relevance/FAISS/Pinecone/NDCG)
- Behavioral signal highlights (response rate, recent activity)

---

## 📊 What the System Catches

| Trap in the dataset | How we handle it |
|---------------------|------------------|
| **Keyword stuffers** (Marketing Manager with all AI skills) | `retrieval_experience` + `product_company_score` look at actual career history |
| **Tier-5 plain-language candidates** | BGE embeddings + retrieval-keyword density in career descriptions |
| **Honeypots** (expert in skill with 0 months) | Multi-pattern detection in `honeypot.py` |
| **Stale candidates** (perfect profile, 6mo inactive) | `active_gate.py` hard filter |
| **Service-only careers** (TCS → Infosys → Wipro) | `product_company_score` multiplicative penalty |
| **Title-chasers** (Senior → Staff → Principal every 1.5y) | `career.py` progression penalty |
| **Senior-junior mismatches** (Director with 3 years) | Honeypot trust score penalty |
| **Too-perfect signal patterns** (synthetic-looking) | Honeypot trust score penalty |

---

## ⚠️ Honesty Notes

- **Embeddings are small (`bge-small-en`)**. We chose this to fit the 5-min/16 GB/CPU constraint. A larger model (`bge-large-en`, `e5-large`) would give better semantic matching but exceed the budget on the grading hardware.
- **No LLM re-ranking**. We deliberately avoided LLM inference during ranking to keep the system deterministic, reproducible, and within the compute budget. The reasoning field is template-based with candidate-specific evidence.
- **BM25 corpus is token-capped** at 8,000 tokens per candidate to keep build time under 5 minutes.

---

## 📈 Future Improvements

1. **Hybrid LLM re-ranking** for top-50 candidates (would require relaxed compute budget)
2. **Learning-to-rank** (XGBoost / LightGBM) trained on recruiter feedback labels
3. **Cross-field consistency checks** — more sophisticated honeypot patterns
4. **Calibration** against held-out historical shortlists
5. **Active learning** — surface low-confidence candidates for human review

---

## 📜 License

Built for the Redrob Hackathon. Code is provided as-is for evaluation purposes.

---

**Built with context, not just keywords.** 🚀
