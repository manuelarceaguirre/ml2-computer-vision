#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-maxvit}"
SEEDS="${TEACHERGRID_SEEDS:-30}"
TEACHER_EPOCHS="${TEACHERGRID_TEACHER_EPOCHS:-15}"
STUDENT_EPOCHS="${TEACHERGRID_STUDENT_EPOCHS:-75}"
BATCH_SIZE="${TEACHERGRID_BATCH_SIZE:-256}"
TEACHER_BATCH_SIZE="${TEACHERGRID_TEACHER_BATCH_SIZE:-16}"
WORKERS="${TEACHERGRID_NUM_WORKERS:-8}"

COMMON=(
  --seeds "$SEEDS"
  --teacher-epochs "$TEACHER_EPOCHS"
  --teacher-head-warmup-epochs 5
  --teacher-head-lr 1e-3
  --teacher-backbone-lr 2e-5
  --teacher-label-smoothing 0.0
  --teacher-augment basic
  --student-epochs "$STUDENT_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --teacher-batch-size "$TEACHER_BATCH_SIZE"
  --num-workers "$WORKERS"
)

case "$MODE" in
  maxvit)
    # First decisive test: full student distillation from the diagnostic-winning
    # MaxViT teacher, without TTA. Run this before spending compute on variants.
    python teachergrid.py \
      --teachers maxvit_base_tf_384.in21k_ft_in1k \
      "${COMMON[@]}" \
      --no-tta
    ;;

  maxvit_tta)
    # Same recipe, but collect 5-view teacher TTA logits. Slower; run only if
    # maxvit mode beats the B0 incumbent or is close.
    python teachergrid.py \
      --teachers maxvit_base_tf_384.in21k_ft_in1k \
      "${COMMON[@]}"
    ;;

  b0_maxvit_committee)
    # Distill one student from averaged B0+MaxViT logits. Individual teachers
    # are trained and diagnosed, but only the committee student is distilled.
    python teachergrid.py \
      --teachers efficientnet_b0,maxvit_base_tf_384.in21k_ft_in1k \
      "${COMMON[@]}" \
      --no-tta \
      --make-committee \
      --committee-only
    ;;

  maxvit_confirm)
    # Multi-seed confirmation for MaxViT no-TTA. Override seeds via env if needed:
    # TEACHERGRID_SEEDS=30,31,32 ./run_maxvit_teachergrid.sh maxvit_confirm
    TEACHERGRID_SEEDS="${TEACHERGRID_SEEDS:-30,31,32}" \
    "$0" maxvit
    ;;

  all)
    "$0" maxvit
    "$0" b0_maxvit_committee
    # TTA is intentionally last because it is slower and may not be needed.
    "$0" maxvit_tta
    ;;

  *)
    cat >&2 <<EOF
Usage: $0 [maxvit|maxvit_tta|b0_maxvit_committee|maxvit_confirm|all]

Environment overrides:
  TEACHERGRID_SEEDS=30,31,32
  TEACHERGRID_TEACHER_EPOCHS=15
  TEACHERGRID_STUDENT_EPOCHS=75
  TEACHERGRID_BATCH_SIZE=256
  TEACHERGRID_TEACHER_BATCH_SIZE=16
  TEACHERGRID_NUM_WORKERS=8
EOF
    exit 2
    ;;
esac
