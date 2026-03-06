# PR restructure summary (MLMaster baseline)

This branch (`mlmaster-baseline`) slims the PR per reviewer feedback.

## What was done

### 1. MLMaster as git submodule
- **Added** `agents/mlmaster/core` as a git submodule pointing to `https://github.com/sjtu-sai-agents/ML-Master`.
- **Removed** all vendored ML-Master code (agent/, interpreter/, backend/, search/, utils/, dataset/, assets/, main_mcts.py, grading_server.py, etc.) so the repo only has integration glue.
- **Kept** in the repo: `config.yaml`, `start.sh`, `Dockerfile`, `additional_notes.txt`, `requirements.txt`.
- **Updated** `Dockerfile` to copy `core/` (submodule) into the image plus the integration files.

### 2. Runs and unrelated changes
- **.gitignore**: Ignore the full `runs/` directory so run outputs (grading JSONs, logs) are not committed.
- **Reverted** `grade_run_group.py` and `run_agent.py` to `main` so shared-infra changes stay in separate PRs.
- **Run scripts**: Replaced hardcoded paths (`/home/amrutharao/...`) with `cd "$(dirname "$0")"` in all `run_*.sh` scripts.

### 3. Bug fixes (reviewer’s list)
- **electricity-demand/grade.py**: Switched from `accuracy_score` to **mean_absolute_error** (regression metric for demand forecasting).
- **playground-series-s5e12/leaderboard.csv**: Replaced the 2-line dummy leaderboard with a valid multi-row CSV (score column, 10 rows).
- **agents/mlmaster/config.yaml**: Added a comment that `time_limit: 7200` is 2 hours (no 360/1hr bug in this file).

### 4. Not in this PR (as requested)
- Run results (`runs/`) — now ignored.
- RD-Agent and AIDE changes — not touched; separate PRs.
- One-off scripts (`fix_aide_submissions.py`, `fix_submission_formats.py`) — not added; left untracked/local.
- Binary assets — removed with vendored code (live in upstream ML-Master).
- Dangling symlink `agents/aide/aideml` — not changed in this PR.

## What you need to do

1. **Push the branch** (from inside `mle-bench`):
   ```bash
   cd mle-bench
   git push -u origin mlmaster-baseline
   ```

2. **Update the parent repo** so its `mle-bench` submodule points at this branch/commit:
   ```bash
   cd /path/to/llms-for-mle-bench
   git submodule set-branch -b mlmaster-baseline mle-bench   # optional: track branch
   cd mle-bench && git checkout mlmaster-baseline && cd ..
   git add mle-bench
   git commit -m "Point mle-bench submodule at mlmaster-baseline branch"
   git push
   ```

3. **Open a new PR** (or replace the old one) with `mlmaster-baseline` and paste the reviewer reply below.

---

## Draft reply to reviewer

You can paste this (or adapt it) in the PR:

---

Thanks for the detailed feedback — this was very helpful.

I’ve restructured the PR as suggested:

- **MLMaster as submodule**: `agents/mlmaster/core` now points to the official [sjtu-sai-agents/ML-Master](https://github.com/sjtu-sai-agents/ML-Master) repo. Only the integration glue remains in-tree: `config.yaml`, `start.sh`, `Dockerfile`, `additional_notes.txt`, and `requirements.txt`. The Dockerfile copies the submodule into the image.
- **Runs**: The entire `runs/` directory is now in `.gitignore`; no run artifacts are committed.
- **Shared infra**: `grade_run_group.py` and `run_agent.py` are reverted to `main`; any shared-infra changes will go in separate PRs.
- **Run scripts**: All `run_*.sh` scripts now use `cd "$(dirname "$0")"` instead of hardcoded `/home/...` paths.
- **Bug fixes**:  
  - **electricity-demand**: Grading uses MAE (regression) instead of accuracy.  
  - **playground-series-s5e12**: Replaced the dummy leaderboard with a proper multi-row `leaderboard.csv`.  
  - **config**: Added a comment that `time_limit: 7200` is 2 hours.

RD-Agent, AIDE config, one-off scripts, and binary assets are unchanged or out of scope for this PR as requested.

This branch is `mlmaster-baseline` in the mle-bench repo. Happy to address any follow-up comments.

---
