# Git Workflow Guide — Zero-Conflict Collaboration

## The Golden Rule
> **Each member works ONLY on their assigned files, in their OWN branch. NEVER push directly to `main`.**

---

## Initial Setup (Everyone does this ONCE)

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/demand-forecasting-agent.git
cd demand-forecasting-agent

# 2. Set up your identity (use your real name)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. Create your feature branch FROM main
git checkout main
git pull origin main
git checkout -b feature/<your-name>-<component>
```

### Branch naming (each member creates ONE branch):

| Member | Branch Name |
|--------|-------------|
| Member 1 | `feature/member1-data` |
| Member 2 | `feature/member2-preprocessing` |
| Member 3 | `feature/member3-model` |
| Member 4 | `feature/member4-api` |
| Member 5 | `feature/member5-frontend` |
| Member 6 | `feature/member6-agent` |
| Member 7 | `feature/member7-integration` |

---

## Daily Workflow (Repeat as you work)

### Step 1: Work on your files
Edit ONLY the files assigned to you. Here's the ownership map:

| Member | Files You CAN Touch | Files You CANNOT Touch |
|--------|--------------------|-----------------------|
| M1 | `data/`, `outputs/` | Everything else |
| M2 | `backend/data_preprocessing.py`, `models/encoders.pkl`, `tests/test_data_preprocessing.py` | Everything else |
| M3 | `notebooks/04_forecasting.ipynb`, `backend/forecasting.py`, `models/best_model.pkl`, `models/README.md`, `outputs/reports/model_evaluation.md`, `tests/test_forecasting.py` | Everything else |
| M4 | `backend/main.py`, `backend/config.py`, `backend/__init__.py`, `requirements.txt`, `.env.example`, `tests/test_api.py` | Everything else |
| M5 | `frontend/app.py` | Everything else |
| M6 | `backend/agent.py`, `backend/inventory.py`, `backend/approval.py`, `notebooks/05_inventory_analysis.ipynb`, `logs/`, `tests/test_inventory.py`, `tests/test_agent.py`, `tests/test_approval.py` | Everything else |
| M7 | `tests/` (integration tests only), `.gitignore`, `README.md`, `ARCHITECTURE.md` | Everything else |

### Step 2: Commit your changes frequently
```bash
# Check what you changed
git status

# Stage your files
git add <your-files-only>
# Example for Member 2:
# git add backend/data_preprocessing.py models/encoders.pkl tests/test_data_preprocessing.py

# Commit with a clear message
git commit -m "M2: implement preprocess_input with categorical encoders"
```

**Commit message format:** `M<number>: <what you did>`

Examples:
- `M1: add data dictionary and EDA figures`
- `M2: implement preprocess_input with FEATURE_ORDER constant`
- `M3: train XGBoost model, serialize best_model.pkl`
- `M4: add FastAPI routes for forecast and approval`
- `M5: build Streamlit form with approval UI`
- `M6: implement agent orchestration with Gemini`
- `M7: add end-to-end integration tests`

### Step 3: Push to YOUR branch
```bash
git push origin feature/<your-name>-<component>
```

---

## Merging to Main (When your work is DONE)

### Option A: Via GitHub Pull Request (RECOMMENDED)

1. Go to the GitHub repo in your browser
2. Click **"Pull requests"** → **"New pull request"**
3. Set:
   - **base:** `main`
   - **compare:** `feature/<your-name>-<component>`
4. Add a title like: `M2: Preprocessing pipeline complete`
5. Add a description of what you did
6. Click **"Create pull request"**
7. Ask **one teammate** to review (or Member 7 reviews all)
8. Once approved, click **"Merge pull request"**
9. **Do NOT delete** your branch until the project is submitted

### Option B: Via Command Line (if you're comfortable)

```bash
# Switch to main and pull latest
git checkout main
git pull origin main

# Merge your branch into main
git merge feature/<your-name>-<component>

# If there are NO conflicts (there shouldn't be if you followed the rules):
git push origin main

# If there ARE conflicts (shouldn't happen):
# STOP. Ask the team. Don't force-push.
```

---

## Merge Order (to avoid conflicts)

Since some members depend on others, merge in this order:

```
1st:  Member 1 (data/)           — no dependencies
2nd:  Member 2 (preprocessing)   — depends on M1's data
3rd:  Member 3 (model)           — depends on M2's encoders
4th:  Member 6 (agent/inventory) — depends on M2, M3
5th:  Member 4 (API)             — depends on M6
6th:  Member 5 (frontend)        — depends on M4
7th:  Member 7 (tests/docs)      — depends on all
```

> **Why this order?** Each member only touches their own files, so technically any order works. But this order ensures each merge builds on the previous one, making it easy to test incrementally.

---

## If Something Goes Wrong

### "I accidentally edited someone else's file"
```bash
# Undo changes to that file (before committing)
git checkout -- <file-you-shouldn't-have-touched>
```

### "I committed to main by mistake"
```bash
# Undo the last commit on main (keeps your changes as unstaged)
git reset HEAD~1

# Now switch to your branch and commit there
git checkout feature/<your-branch>
git add .
git commit -m "M<N>: your message"
```

### "I have a merge conflict"
This should NOT happen if everyone follows the rules. But if it does:

1. **Don't panic.** Read the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
2. **Talk to the teammate** whose code conflicts with yours
3. **Decide together** which version to keep
4. Edit the file, remove conflict markers
5. `git add <file>` → `git commit` → `git push`

### "I need a file that another member hasn't pushed yet"
```bash
# Pull their branch to see their work
git fetch origin
git checkout feature/<their-branch>
# Look at what you need, take notes
git checkout feature/<your-branch>
# Continue your work
```

---

## Quick Reference

```bash
# See all branches
git branch -a

# Switch to your branch
git checkout feature/<your-branch>

# Get latest main
git checkout main && git pull origin main

# Update your branch with latest main (if needed)
git checkout feature/<your-branch>
git merge main

# See commit history
git log --oneline -10

# See what files changed
git diff --name-only main
```

---

## Checklist Before Submitting (Sep 14)

- [ ] All 7 feature branches merged to `main`
- [ ] `main` branch has all files from the folder structure
- [ ] `pytest tests/ -v` passes
- [ ] Backend starts: `uvicorn backend.main:app --reload --port 8000`
- [ ] Frontend starts: `streamlit run frontend/app.py`
- [ ] Forecast works end-to-end
- [ ] Approval flow works (approve/modify/reject)
- [ ] `logs/approval_log.json` records decisions
- [ ] README.md is accurate
- [ ] No `.env` files committed (only `.env.example`)
- [ ] No `__pycache__/` or `.ipynb_checkpoints/` committed
