#!/usr/bin/env bash
# Full generative-sampler benchmark (n = 5000 per batch by default). ~2h on 4 CPU cores.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results_gen}
REPS=${REPS:-5}
for suite in steps quality covshift concept; do
  python3 -m gen_fsds.run_benchmark --suite $suite --out $OUT --reps $REPS --jobs 4
done
python3 -m gen_fsds.report $OUT
