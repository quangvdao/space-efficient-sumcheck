#!/bin/sh

# We measure (i) wall time; and (ii) maximum resident set size, using the GNU-time facility.
# NOTE: CLI order is (algorithm_label, field_label, num_variables, stage_size [,d]).
#       This differs from the printed usage string.

# Optional toggles:
#   ./run_benches.sh                       → run all fields (Field64 Field128 FieldBn254)
#   ./run_benches.sh --bn254-only          → restrict to FieldBn254 only
#   ./run_benches.sh simple                → run one point (n=16 by default) for each algo/field
#   ./run_benches.sh --d 8                 → set number of multilinears (product benches only)
#   Toggles can be combined, e.g.: ./run_benches.sh --bn254-only --d 8 simple

BN254_ONLY=0
SIMPLE=0
D_OVERRIDE=""
# Parse optional toggles in any order
while [ $# -gt 0 ]; do
  case "$1" in
    --bn254-only)
      BN254_ONLY=1;
      shift;;
    simple|--simple)
      SIMPLE=1;
      shift;;
    --d)
      shift;
      if [ -n "$1" ]; then
        D_OVERRIDE="$1";
        shift;
      fi;;
    *)
      break;;
  esac
done

# Ensure release binary exists
BIN=./target/release/sumcheck-benches
if [ ! -x "$BIN" ]; then
  echo "Building release bench binary..."
  cargo build -p sumcheck-benches --release || { echo "Build failed"; exit 1; }
fi

# Limit to BN254 only and include the improved-time product prover
# Commented out multilinear algorithms: `Blendy1 Blendy2 VSBW Blendy3 Blendy4 CTY`
algorithms="ProductBlendy2 ProductVSBW ProductCTY ProductImprovedTime"
if [ $BN254_ONLY -eq 1 ]; then
  fields="FieldBn254"
else
  fields="Field64 Field128 FieldBn254"
fi

for algorithm in $algorithms; do
    for field in $fields; do
        # variable sweep
        if [ $SIMPLE -eq 1 ]; then
          num_vars=20
          end_vars=20
        else
          num_vars=16
          end_vars=30
        fi
        while [ $num_vars -le $end_vars ]; do
            case "$algorithm" in
                # "Blendy1") stage_size="1" ;;
                # "Blendy2") stage_size="2" ;;
                # "Blendy3") stage_size="3" ;;
                # "Blendy4") stage_size="4" ;;
                # "VSBW") stage_size="1" ;;
                # "CTY") stage_size="1" ;;
                "ProductBlendy2") stage_size="2" ;;
                "ProductVSBW") stage_size="1" ;;
                "ProductCTY") stage_size="1" ;;
                "ProductImprovedTime") stage_size="1" ;;
                *) ;;
            esac
            case "$algorithm" in
                # "Blendy1") algorithm_label="Blendy" ;;
                # "Blendy2") algorithm_label="Blendy" ;;
                # "Blendy3") algorithm_label="Blendy" ;;
                # "Blendy4") algorithm_label="Blendy" ;;
                # "VSBW") algorithm_label="VSBW" ;;
                # "CTY") algorithm_label="CTY" ;;
                "ProductBlendy2") algorithm_label="ProductBlendy" ;;
                "ProductVSBW") algorithm_label="ProductVSBW" ;;
                "ProductCTY") algorithm_label="ProductCTY" ;;
                "ProductImprovedTime") algorithm_label="ProductImprovedTime" ;;
                *) ;;
            esac
            # NOTE FOR NEXT LINE: mac with gnu-time installed → "gtime"; otherwise fall back to "/usr/bin/time -l"
            # append optional d only for product algorithms
            D_ARG=""
            case "$algorithm_label" in
              Product*)
                if [ -n "$D_OVERRIDE" ]; then D_ARG=" $D_OVERRIDE"; fi;;
            esac
            if command -v gtime >/dev/null 2>&1; then
              output=`(gtime -v "$BIN" $algorithm_label $field $num_vars $stage_size$D_ARG) 2>&1`
              user_time_seconds=$(echo "$output" | awk -F': ' '/User time \(seconds\)/{print $2; exit}')
              ram_kilobytes=$(echo "$output" | awk -F': ' '/Maximum resident set size \(kbytes\)/{print $2; exit}')
            else
              output=`(/usr/bin/time -l "$BIN" $algorithm_label $field $num_vars $stage_size$D_ARG) 2>&1`
              # BSD time output parsing (best-effort)
              user_time_seconds=$(echo "$output" | awk '/[[:space:]]user[[:space:]]/{print $(NF-3); exit}')
              ram_kilobytes=$(echo "$output" | awk -F': ' '/maximum resident set size/{print $2; exit}')
            fi
            user_time_ms=$(awk "BEGIN {printf \"%.0f\", $user_time_seconds * 1000}")
            ram_bytes=$(awk "BEGIN {printf \"%.0f\", $ram_kilobytes * 1000}")
            echo "$algorithm, $field, $num_vars, $user_time_ms, $num_vars, $ram_bytes"
            if [ $SIMPLE -eq 1 ]; then
              break
            else
              num_vars=$((num_vars + 2))
            fi
        done
    done
done

# NOTE: helpful Unix commands
#
# 1) You can run this shell in the background while piping the output to a file like so:
#   nohup ./run_benches.sh &> output_file.txt &
#
# 2) If you need to kill the running process you can find pid with:
#   lsof | grep output_file
#  Then:
#     kill <pid>
