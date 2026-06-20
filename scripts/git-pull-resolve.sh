#!/usr/bin/env bash
set -euo pipefail

# Simple interactive helper to resolve "Need to specify how to reconcile divergent branches"
# Usage:
#   bash scripts/git-pull-resolve.sh
# or (recommended)
#   chmod +x scripts/git-pull-resolve.sh
#   ./scripts/git-pull-resolve.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Git Pull Resolve Helper =="
echo "Repository: $(basename "$REPO_ROOT")"
echo

# Show brief branch status and divergence hint
git status --branch --short
echo

# Show recent history to help decision making
echo "Recent commits (local vs remote):"
git for-each-ref --format='%(refname:short) %(upstream:short)' refs/heads | while read -r local upstream; do
  if [ -n "$upstream" ]; then
    echo "Branch: $local (tracking $upstream)"
  fi
done
echo
echo "Tip: You can inspect the commit graph with: git log --oneline --graph --decorate --all"
echo

# Current pull configuration
CUR_PULL_REBASE=$(git config --get pull.rebase || echo "<unset>")
CUR_PULL_FF=$(git config --get pull.ff || echo "<unset>")
echo "Current git pull configuration:"
echo "  pull.rebase = $CUR_PULL_REBASE"
echo "  pull.ff     = $CUR_PULL_FF"
echo

cat <<'EOF'
You have divergent branches. Choose how to reconcile for this pull:
  1) Merge (create a merge commit)    -> git pull --no-rebase
  2) Rebase (replay local commits)    -> git pull --rebase
  3) Fast-forward only (refuse merges)-> git pull --ff-only
  4) Set a global default and perform the chosen action
  5) Show help / abort
EOF

read -rp "Select option [1-5]: " OPT

case "$OPT" in
  1)
    echo "Executing: git pull --no-rebase origin main"
    git pull --no-rebase origin main
    ;;
  2)
    echo "Executing: git pull --rebase origin main"
    git pull --rebase origin main
    ;;
  3)
    echo "Executing: git pull --ff-only origin main"
    git pull --ff-only origin main
    ;;
  4)
    echo "Choose default to set globally:"
    echo "  a) pull.rebase=false (merge)"
    echo "  b) pull.rebase=true  (rebase)"
    echo "  c) pull.ff=only      (fast-forward only)"
    read -rp "Select default [a-c]: " DEF
    case "$DEF" in
      a)
        echo "Setting global: git config --global pull.rebase false"
        git config --global pull.rebase false
        echo "Now performing git pull --no-rebase origin main"
        git pull --no-rebase origin main
        ;;
      b)
        echo "Setting global: git config --global pull.rebase true"
        git config --global pull.rebase true
        echo "Now performing git pull --rebase origin main"
        git pull --rebase origin main
        ;;
      c)
        echo "Setting global: git config --global pull.ff only"
        git config --global pull.ff only
        echo "Now performing git pull --ff-only origin main"
        git pull --ff-only origin main
        ;;
      *)
        echo "Invalid choice. Aborting."
        exit 1
        ;;
    esac
    ;;
  5)
    echo "Aborting. No changes made."
    exit 0
    ;;
  *)
    echo "Invalid selection. Exiting."
    exit 1
    ;;
esac

echo
echo "Done. Inspect repository state with: git status"
echo "If conflicts occurred during rebase/merge, resolve files, then run:"
echo "  git add <resolved-files>"
echo "  git rebase --continue   # if rebasing"
echo "  git commit              # if merging and conflicts were resolved"
