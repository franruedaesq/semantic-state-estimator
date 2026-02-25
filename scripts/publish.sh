#!/usr/bin/env bash
# scripts/publish.sh
# One-command build → test → publish pipeline for semantic-state-estimator.
# Usage:
#   npm run release              # interactive prompt before publishing
#   DRY_RUN=1 npm run release    # always does a dry-run, never publishes

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACKAGE_NAME=$(node -p "require('./package.json').name")
PACKAGE_VERSION=$(node -p "require('./package.json').version")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📦  $PACKAGE_NAME @ v$PACKAGE_VERSION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Type-check ──────────────────────────────────
echo ""
echo "🔍 [1/4] Type-checking..."
npm run lint

# ── Step 2: Tests ──────────────────────────────────────
echo ""
echo "🧪 [2/4] Running tests..."
npm run test

# ── Step 3: Build ──────────────────────────────────────
echo ""
echo "🔨 [3/4] Building dist/..."
npm run build

# ── Step 4: Dry-run (always) ───────────────────────────
echo ""
echo "📋 [4/4] Dry-run — files that would be published:"
npm publish --dry-run

# ── Early exit for dry-run mode ────────────────────────
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""
  echo "✅  DRY_RUN=1 — skipping live publish."
  exit 0
fi

# ── Confirm before publishing ──────────────────────────
echo ""
echo "⚠️  About to publish $PACKAGE_NAME@$PACKAGE_VERSION to the npm registry."
read -r -p "   Type 'yes' to continue: " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
  echo "   Aborted."
  exit 1
fi

# ── Publish ────────────────────────────────────────────
echo ""
echo "🚀 Publishing..."
npm publish

echo ""
echo "✅  Successfully published $PACKAGE_NAME@$PACKAGE_VERSION"
echo "   https://www.npmjs.com/package/$PACKAGE_NAME"
