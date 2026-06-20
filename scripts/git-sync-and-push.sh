#!/usr/bin/env bash
set -euo pipefail

# Simple interactive helper to resolve divergent branches and push safely.
# Usage: bash scripts/git-sync-and-push.sh
# Make executable: chmod +x scripts/git-sync-and-push.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Git Sync & Push Helper — repository: $(basename "$REPO_ROOT")"
echo

# Show current branch and divergence summary
git status --short --branch
echo

# Fetch remote updates
echo "Fetching from origin..."
git fetch origin
echo

# Try fast-forward pull first (non-destructive)
echo "Attempting fast-forward merge..."
if git pull --ff-only origin main; then
  echo "Fast-forward successful. Now pushing to origin/main..."
  git push -u origin main
  echo "Push complete."
  exit 0
else
  echo
  echo "Fast-forward not possible — branches have diverged."
  echo "Choose how to reconcile:"
  echo "  1) Merge (create merge commit)"
  echo "  2) Rebase (replay local commits on top of remote)"
  echo "  3) Inspect differences and abort"
  echo "  4) Abort"
  read -rp "Select option [1-4]: " CHOICE
fi

case "$CHOICE" in
  1)
    echo "Performing merge: git pull --no-rebase origin main"
    if git pull --no-rebase origin main; then
      echo "Merge completed. Now pushing..."
      git push -u origin main
      echo "Push complete."
    else
      echo "Merge failed - please resolve conflicts, stage files and run 'git commit' then 'git push'."
      exit 1
    fi
    ;;
  2)
    echo "Performing rebase: git pull --rebase origin main"
    if git pull --rebase origin main; then
      echo "Rebase completed. Now pushing..."
      git push -u origin main
      echo "Push complete."
    else
      echo "Rebase encountered conflicts. To continue after resolving conflicts:"
      echo "  git add <resolved-files>"
      echo "  git rebase --continue"
      echo "Or to abort rebase and return to previous state:"
      echo "  git rebase --abort"
      exit 1
    fi
    ;;
  3)
    echo "Remote commits not merged locally. Showing divergence summary..."
    echo
    echo "Commits on remote not in local:"
    git log --oneline --no-decorate --left-right origin/main...main | sed -n '1,20p'
    echo
    echo "Commits on local not in remote:"
    git log --oneline --no-decorate main...origin/main | sed -n '1,20p'
    echo
    echo "Review the above (use git log --oneline --graph --decorate --all for more). Aborting helper."
    exit 0
    ;;
  4)
    echo "Aborting. No changes made."
    exit 0
    ;;
  *)
    echo "Invalid option. Exiting."
    exit 1
    ;;
esac

echo "Done."
