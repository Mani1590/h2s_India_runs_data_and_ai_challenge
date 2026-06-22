#!/usr/bin/env bash
set -e

echo "=== Phase 1: build offline artifacts (run ONCE) ==="
python scripts/build_artifacts.py

echo "=== Phase 2: generate submission (5-min budget) ==="
python src/generate_submission.py

echo "=== Validate ==="
python validate_submission.py output/submission.csv
