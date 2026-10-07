#!/bin/zsh
# The temporal-kernel gate, as benchmarks/temporal_gate_preregistration.md sets it out: section 11's
# commands in its order (kernel level, then engine level; runtime A, then runtime B), then the
# plane-clearing pass of section 9, each in its own process. From the repository root, on the
# prepared machine (section 3):
#
#   caffeinate -dimsu benchmarks/temporal_gate.sh <results> <runtime-B python>
#
# <results> is a new directory under a gitignored path. Measurement stops at the first command that
# fails or document whose result checks fail (section 10); nothing is re-run here. Every step's
# output goes to <results>/<step>.log, and the progress to <results>/progress.txt.
set -u
if (( $# != 2 )); then
  print -u2 "usage: $0 <results> <runtime-B python>"
  exit 2
fi
results=$1
python_b=$2
if [[ -e $results ]]; then
  print -u2 "$results exists; results are never overwritten"
  exit 2
fi
if ! git diff --quiet HEAD || [[ -n $(git ls-files --others --exclude-standard) ]]; then
  print -u2 "the working tree is not clean"
  exit 2
fi
mkdir -p $results
{ git rev-parse HEAD; git rev-parse HEAD:src; } > $results/measured.txt

step() {
  local name=$1
  shift
  print "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $name" >> $results/progress.txt
  "$@" > $results/$name.log 2>&1
  local exit_status=$?
  print "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $name exit $exit_status" >> $results/progress.txt
  if (( exit_status != 0 )); then
    print "STOPPED: $name failed (exit $exit_status); see $results/$name.log" >> $results/progress.txt
    exit 1
  fi
}

checked() {
  # Stops measurement if any run of the document failed its result checks (section 10).
  local document=$1
  uv run --frozen python -c '
import json, sys
document = json.load(open(sys.argv[1]))
failed = [c["kernel"] for c in document["cells"] for checks in c.get("checks") or [] if not checks.get("valid")]
sys.exit(f"{len(failed)} runs failed their result checks" if failed else 0)
' $document >> $results/progress.txt 2>&1 || { print "STOPPED: $document failed result checks" >> $results/progress.txt; exit 1; }
}

for runtime in A B; do
  if [[ $runtime == A ]]; then py=(uv run --frozen python); else py=($python_b); fi
  for level in kernel engine; do
    step ${runtime}_${level} $py -m benchmarks run --suite temporal-gate --target temporal-$level \
      --out $results/${runtime}_${level}.json
    checked $results/${runtime}_${level}.json
  done
  for level in kernel engine; do
    step ${runtime}_${level}_stage2 $py -m benchmarks stage2 $results/${runtime}_${level}.json \
      --out $results/${runtime}_${level}_stage2.json
    checked $results/${runtime}_${level}_stage2.json
  done
  step ${runtime}_gate $py -m benchmarks gate --kernel-level $results/${runtime}_kernel.json \
    --engine-level $results/${runtime}_engine.json \
    --kernel-stage2 $results/${runtime}_kernel_stage2.json \
    --engine-stage2 $results/${runtime}_engine_stage2.json \
    --out $results/${runtime}_gate.json
done
for runtime in A B; do
  if [[ $runtime == A ]]; then py=(uv run --frozen python); else py=($python_b); fi
  step ${runtime}_planes $py -m benchmarks run --suite temporal-gate --target temporal-planes --runs 1 \
    --memory-calls 0 --out $results/${runtime}_planes.json
  checked $results/${runtime}_planes.json
done
print "$(date -u +%Y-%m-%dT%H:%M:%SZ) DONE" >> $results/progress.txt
