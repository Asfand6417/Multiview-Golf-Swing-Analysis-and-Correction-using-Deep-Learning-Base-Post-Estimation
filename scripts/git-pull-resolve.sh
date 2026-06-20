#!/usr/bin/env bash
set -euo pipefail
# Simple interactive helper to resolve "Need to specify how to reconcile divergent branches"
# Usage: bash scripts/git-pull-resolve.sh
# Make executable: chmod +x scripts/git-pull-resolve.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Git Pull Resolve Helper — repository: $(basename "$REPO_ROOT")"
echo

# Show current status and divergence summary
git status --short --branch
echo

# Provide current config for pull behavior
CURRENT_PULL_REBASE=$(git config --get pull.rebase || echo "<unset>")
CURRENT_PULL_FF=$(git config --get pull.ff || echo "<unset>")

echo "Current git pull settings:"
echo "  pull.rebase = $CURRENT_PULL_REBASE"
echo "  pull.ff     = $CURRENT_PULL_FF"
echo

echo "You have divergent branches if your local and remote have different commits."
echo "Choose a reconciliation strategy for this pull (one-off):"
echo "  1) Merge (create a merge commit)    -> git pull --no-rebase"
echo "  2) Rebase (replay local commits)    -> git pull --rebase"
echo "  3) Fast-forward only (refuse merge) -> git pull --ff-only"
echo "  4) Set a global default and perform the chosen action"
echo "  5) Abort (do nothing)"
echo
read -rp "Select option [1-5]: " OPT

case "$OPT" in
  1)
    echo "Performing: git pull --no-rebase origin main"
    git pull --no-rebase origin main
    ;;
  2)
    echo "Performing: git pull --rebase origin main"
    git pull --rebase origin main
    ;;
  3)
    echo "Performing: git pull --ff-only origin main"
    git pull --ff-only origin main
    ;;
  4)
    echo "Choose default to set globally:"
    echo "  a) Set pull.rebase=false (merge)"
    echo "  b) Set pull.rebase=true (rebase)"
    echo "  c) Set pull.ff=only (fast-forward only)"
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
echo "Pull completed. Run 'git status' or 'git log --oneline --graph --decorate --all' to inspect."
