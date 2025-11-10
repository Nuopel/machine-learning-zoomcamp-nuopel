# Wine Price Prediction - ML Midterm Project

## Project Context

**Objective:** Build an end-to-end ML system that predicts French red wine prices (EUR) from Vivino data, deploy as REST API with Docker.

**Timeline:** (2025-11-10 to 2025-11-18)
**Status:** 🟢 Execution Phase (M1: Foundation)

---

## Problem Definition

**What:** Predict wine price (EUR)  characteristic

**Why:** Helps consumers identify fairly-priced wines; demonstrates ML pipeline competency

**How:** Regression model (Linear → Random Forest → XGBoost)  → Docker

**Success:** 

---

## Work Breakdown (5 Lots)

| #   | Lot           | Duration | Target | Status         |
| --- | ------------- | -------- | ------ | -------------- |
| 1   | Setup & Data  |          | Nov 10 | 🔄 IN PROGRESS |
| 2   | EDA           |          | Nov    | ⏳ PENDING      |
| 3   | Models        |          | Nov    | ⏳ PENDING      |
| 4   | API           |          | Nov    | ⏳ PENDING      |
| 5   | Docker & Docs |          | Nov 18 | ⏳ PENDING      |

---

## Quick Start

### READ FIRST

1. This README (you're reading it)
2. `wbs/LOT-001.md` — See what LOT-1 delivers
3. `wbs/STATUS.md` — Check current status anytime

### THEN EXECUTE

1. Follow `wbs/LOT-001.md` step-by-step
2. Create: `data/wine_data_cleaned.csv` + `requirements.txt`
3. Commit changes, update `wbs/STATUS.md`

### REPEAT FOR LOTS 2-5

Each lot: Read spec → Execute checklist → Update status

---

## Deliverables Overview

### LOT-001: Project Setup & Data Preparation

- Repository structure (src/, tests/, data/, models/, docs/)
- Vivino CSV loaded and cleaned
- requirements.txt with pinned versions
- **.gitignore configured**

### LOT-002: EDA & Feature Analysis

- Notebook with exploratory analysis
- Visualizations (correlations, distributions)
- Feature insights documented

### LOT-003: Model Training & Selection

- 3 trained models (Linear, RF, XGBoost)
- Model comparison table (RMSE, MAE, R²)
- Best model saved (best_model.pkl)
- train.py script exported

### LOT-004: Web Service & Packaging

- Flask app (serve.py) on port 9696
- /predict endpoint (POST with JSON)
- /health endpoint (GET monitoring)
- predict.py inference logic

### LOT-005: Docker & Documentation

- Dockerfile (python:3.11-slim)
- README with API examples
- Unit tests (≥80% coverage)
- Code quality (black, ruff, mypy, pytest all passing)

---

## Project Structure

```
Midterm/
├── README.md                         ← You are here
├── wbs/                              ← Work packages folder
│   ├── LOT-001.md                   ← Specification & checklist for LOT-1
│   ├── LOT-002.md                   ← Specification & checklist for LOT-2
│   ├── LOT-003.md                   ← Specification & checklist for LOT-3
│   ├── LOT-004.md                   ← Specification & checklist for LOT-4
│   ├── LOT-005.md                   ← Specification & checklist for LOT-5
│   └── STATUS.md                    ← Current project status (update after each LOT)
│
├── docs/
│   └── standards.md                 ← Code style, testing, git workflow
│
├── src/                             ← Python modules (created in LOTs)
├── tests/                           ← Unit tests (created in LOTs)
├── data/                            ← Datasets (created in LOT-001)
├── models/                          ← Model artifacts (created in LOT-003)
│
├── .gitignore                       ← Created in LOT-001
├── requirements.txt                 ← Created in LOT-001
├── Dockerfile                       ← Created in LOT-005
└── notebook.ipynb                   ← Created in LOT-002
```

---

## Quality Standards

Before committing any code:

```bash
black src/ tests/          # Format (88 chars)
ruff check src/ tests/     # Lint (strict)
mypy src/                  # Type check
pytest tests/ --cov        # Test (≥80% coverage)
```

See `docs/standards.md` for details.

---

## Milestones & Key Dates

| Date       | Milestone      | LOTs | Deliverable              |
| ---------- | -------------- | ---- | ------------------------ |
| 2025-11-10 | M0: Planning ✅ | —    | Project defined          |
| 2025-11-14 | M1: Foundation | 1-2  | Data ready, EDA complete |
| 2025-11-17 | M2: Models     | 3    | Best model selected      |
| 2025-11-19 | M3: Deployment | 4    | API running              |
| 2025-11-21 | M4: Release    | 5    | Docker works, tests pass |
| 2025-11-24 | M5: Submit     | —    | Repository ready         |

---

## Execution Workflow

### For Each LOT

1. **Read:** `wbs/LOT-00X.md` (understand objective & criteria)
2. **Execute:** Follow checklist step-by-step
3. **Verify:** All acceptance criteria checked
4. **Commit:** `git commit -m "Complete LOT-00X: [description]"`
5. **Update:** `wbs/STATUS.md` with completion date
6. **Start next:** Proceed to LOT-00X+1

---

## Status & Progress

**Check anytime:** `wbs/STATUS.md`

Shows:

- Which LOTs are in_progress, pending, blocked
- Current milestone & days remaining
- Next action

---

## Important Files

| File               | Purpose                        | When             |
| ------------------ | ------------------------------ | ---------------- |
| **README.md**      | This file — context & overview | Start here       |
| **wbs/LOT-00X.md** | Lot objective & checklist      | Execute each LOT |
| **wbs/STATUS.md**  | Current status                 | Track progress   |
|                    |                                |                  |

---



---

## Next Steps

1. ✅ Read this README (done)
2. → Read `wbs/LOT-001.md`
3. → Follow checklist in LOT-001.md
4. → Complete by 2025-11-12
5. → Update `wbs/STATUS.md`
6. → Proceed to LOT-002

---

**Start:** 2025-11-10
**Target:** 2025-11-18
