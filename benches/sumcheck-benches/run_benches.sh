#!/bin/sh

# We measure (i) wall time; and (ii) maximum resident set size, using the GNU-time facility.
# NOTE: CLI order is (algorithm_label, field_label, num_variables, stage_size [,d]).
#       This differs from the printed usage string.

# Optional toggles:
#   ./run_benches.sh                             → run all fields (Field64 Field128 FieldBn254)
#   ./run_benches.sh --bn254-only                → restrict to FieldBn254 only
#   ./run_benches.sh simple                      → run one point (n=16 by default) for each algo/field
#   ./run_benches.sh --d 8                       → set number of multilinears (product benches only)
#   ./run_benches.sh --compare-time              → compare only ProductVSBW vs ProductImprovedTime (BN254 enforced)
#   Toggles can be combined, e.g.: ./run_benches.sh --compare-time --d 8 simple

BN254_ONLY=0
COMPARE_TIME_ONLY=0
SIMPLE=0
D_OVERRIDE=""
NO_BUILD=0
# Resolve script directory and ensure we run relative to it so paths work from any CWD
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || { echo "Failed to cd to script dir: $SCRIPT_DIR"; exit 1; }
# Parse optional toggles in any order
while [ $# -gt 0 ]; do
  case "$1" in
    --bn254-only)
      BN254_ONLY=1;
      shift;;
    --compare-time|--time-vs-improved)
      COMPARE_TIME_ONLY=1;
      BN254_ONLY=1; # ImprovedTime is BN254-only
      shift;;
    simple|--simple)
      SIMPLE=1;
      shift;;
    --d)
      shift;
      if [ -n "$1" ]; then
        D_OVERRIDE="$1";
        # Validate d value early
        case "$D_OVERRIDE" in
          2|3|4|8|16|32) ;;
          *) 
            echo "Error: Unsupported d value: $D_OVERRIDE. Product algorithms only support d ∈ {2, 3, 4, 8, 16, 32}"
            exit 1;;
        esac
        shift;
      fi;;
    --no-build)
      NO_BUILD=1;
      shift;;
    *)
      break;;
  esac
done

# Build the release bench binary unless disabled
BIN="$SCRIPT_DIR/target/release/sumcheck-benches"
if [ $NO_BUILD -ne 1 ]; then
  echo "Building release bench binary..."
  cargo build -p sumcheck-benches --release || { echo "Build failed"; exit 1; }
fi
if [ ! -x "$BIN" ]; then
  echo "Error: bench binary not found at $BIN"
  exit 1
fi

# Select algorithm set
if [ $COMPARE_TIME_ONLY -eq 1 ]; then
  algorithms="ProductVSBW ProductImprovedTime"
else
  # Include the improved-time product prover alongside others by default
  # Commented out multilinear algorithms: `Blendy1 Blendy2 VSBW Blendy3 Blendy4 CTY`
  algorithms="ProductBlendy2 ProductVSBW ProductCTY ProductImprovedTime"
fi
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
              exit_code=$?
              if [ $exit_code -ne 0 ]; then
                echo "ERROR: Benchmark failed for $algorithm_label $field $num_vars $stage_size$D_ARG"
                echo "Output: $output"
                exit $exit_code
              fi
              user_time_seconds=$(echo "$output" | awk -F': ' '/User time \(seconds\)/{print $2; exit}')
              ram_kilobytes=$(echo "$output" | awk -F': ' '/Maximum resident set size \(kbytes\)/{print $2; exit}')
            else
              output=`(/usr/bin/time -l "$BIN" $algorithm_label $field $num_vars $stage_size$D_ARG) 2>&1`
              exit_code=$?
              if [ $exit_code -ne 0 ]; then
                echo "ERROR: Benchmark failed for $algorithm_label $field $num_vars $stage_size$D_ARG"
                echo "Output: $output"
                exit $exit_code
              fi
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
