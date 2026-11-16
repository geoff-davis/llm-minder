#!/usr/bin/env bash
# prek hook: run the markdownlint-cli2 pinned in package-lock.json on the given
# files. Skips (exit 0, with a message) when node_modules is absent, so a clone
# without Node tooling can still commit; CI's quality job enforces the lint.
# Prints nothing on success, because the hook runs with `verbose: true` to show
# the skip message.
set -u

if [ ! -x node_modules/.bin/markdownlint-cli2 ]; then
  echo "markdownlint-cli2 skipped: node_modules is absent (run 'npm ci' to enable it)."
  exit 0
fi

if output=$(npx --no-install markdownlint-cli2 "$@" 2>&1); then
  exit 0
else
  status=$?
  printf '%s\n' "$output"
  exit "$status"
fi
