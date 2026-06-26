#!/usr/bin/env bash
# Remove "Co-authored-by: Cursor <cursoragent@cursor.com>" from all commit messages.
# Usage: ./scripts/strip_cursor_coauthor.sh [--push origin main]
set -euo pipefail

cd "$(dirname "$0")/.."

TRAILER='^Co-authored-by: Cursor <cursoragent@cursor.com>$'

echo "Commits with Cursor co-author trailer:"
git log --format='%h %s' --grep='Co-authored-by: Cursor' 2>/dev/null || true
if git log --format='%B' | grep -q "$TRAILER"; then
  echo "(found in commit bodies)"
else
  echo "(none in bodies — may still run filter to be sure)"
fi

read -r -p "Rewrite ALL commit messages to strip Cursor trailer? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 0; }

FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter "sed '/${TRAILER}/d'" \
  -- --all

echo "=== verify ==="
if git log --format='%B' | grep -qi 'co-authored-by: cursor'; then
  echo "ERROR: trailer still present"
  exit 1
fi
echo "Clean."

if [[ "${1:-}" == "--push" ]]; then
  shift
  remote="${1:-origin}"
  branch="${2:-main}"
  read -r -p "Force-push to ${remote}/${branch}? [y/N] " ans2
  [[ "${ans2,,}" == "y" ]] || { echo "Skipped push."; exit 0; }
  git push --force-with-lease "$remote" "$branch"
  echo "Pushed."
fi
