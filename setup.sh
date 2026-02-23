# create_repo_from_model.sh
# Usage:
#   1) cd to the folder containing your echo_gnn_v3.py
#   2) paste and run: bash <(curl -sS https://gist.githubusercontent.com/...)  # if hosted
# Or copy the block below and save as create_repo_from_model.sh then run: bash create_repo_from_model.sh --remote <git-url> [--push]
#
# This script will NOT modify echo_gnn_v3.py
set -euo pipefail

REMOTE_URL=""
DO_PUSH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE_URL="$2"; shift 2;;
    --push) DO_PUSH=true; shift;;
    --help) echo "Usage: bash create_repo_from_model.sh --remote <git-url> [--push]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

REPO_DIR="$(pwd)"
MODEL_FILE="echo_gnn_v3.py"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Error: $MODEL_FILE not found in $REPO_DIR"
  echo "Place your echo_gnn_v3.py here and re-run."
  exit 2
fi

echo "Preparing repository in: $REPO_DIR"
echo "Model file preserved: $MODEL_FILE"

# 1) create minimal files (do NOT modify model file)
if [[ ! -f README.md ]]; then
  cat > README.md <<'MD'
# ECHO-GNN (research snapshot)

This repository contains the research implementation file `echo_gnn_v3.py` (research release v3.0.0).

**Do not modify** `echo_gnn_v3.py` if preserving the original release is important.

Citation: see `echo_gnn_v3.py` for author and version. :contentReference[oaicite:0]{index=0}
MD
  echo "Wrote README.md"
else
  echo "README.md already exists; left untouched."
fi

if [[ ! -f LICENSE ]]; then
  cat > LICENSE <<'L'
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
Copyright 2026 Emilio Ferrara
L
  echo "Wrote LICENSE"
fi

if [[ ! -f .gitignore ]]; then
  cat > .gitignore <<'GI'
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
.env

# Data / results
*.npz
*.npy
*.pth
*.pt
*.zip

# System
.DS_Store
GI
  echo "Wrote .gitignore"
fi

# 2) initialize git repo if needed
if [ ! -d .git ]; then
  git init
  echo "Initialized empty git repo"
else
  echo "Git repo already initialized"
fi

# 3) make initial commit (only add new files and model file)
git add "$MODEL_FILE" README.md LICENSE .gitignore >/dev/null 2>&1 || true
if git diff-index --quiet HEAD --; then
  echo "No changes to commit (working tree clean)"
else
  git commit -m "chore: initial import of echo_gnn_v3.py (research snapshot v3.0.0)"
  echo "Committed model and minimal repo files"
fi

# 4) create an annotated tag if not existing
if git rev-parse "refs/tags/v3.0.0" >/dev/null 2>&1; then
  echo "Tag v3.0.0 already exists"
else
  git tag -a v3.0.0 -m "ECHO v3.0.0 — research snapshot (model file only)"
  echo "Created tag v3.0.0"
fi

# 5) optionally add remote
if [[ -n "$REMOTE_URL" ]]; then
  if git remote | grep -q origin; then
    git remote set-url origin "$REMOTE_URL"
  else
    git remote add origin "$REMOTE_URL"
  fi
  echo "Remote origin set to $REMOTE_URL"
fi

# 6) optionally push
if [[ "$DO_PUSH" = true ]]; then
  if [[ -z "$REMOTE_URL" ]]; then
    echo "Cannot push: --push was set but no --remote provided"
    exit 3
  fi
  echo "Pushing main branch and tags to origin..."
  git branch -M main || true
  git push -u origin main
  git push origin --tags
  echo "Pushed to remote"
else
  echo "Not pushing. To push, re-run with: bash create_repo_from_model.sh --remote <git-url> --push"
fi

echo "Done. Repo ready to review and push."
