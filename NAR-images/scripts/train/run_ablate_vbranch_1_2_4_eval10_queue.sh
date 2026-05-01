#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

for script in \
  scripts/train/ablate_vbranch_last1_eval10.sh \
  scripts/train/ablate_vbranch_last2_eval10.sh \
  scripts/train/ablate_vbranch_last4_eval10.sh; do
  echo "===== START ${script} $(date) ====="
  bash "${script}"
  echo "===== DONE  ${script} $(date) ====="
done
