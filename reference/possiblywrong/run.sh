#!/usr/bin/env bash
# Build and run possiblywrong's retrograde solver (https://github.com/possibly-wrong/gobblet)
# as an independent reference. It writes the full value table as
# gobblet_<num_sizes>_<num_per_size>_<allow_move>.dat (a 4 GiB hash table).
#
# After solving, the original program enters an interactive loop that spins forever once stdin
# is exhausted, so `game.play();` in main is replaced by `return 0;`.
#
# Usage: run.sh <output dir> [num_sizes num_per_size allow_move]   (default: 3 2 1)
set -euo pipefail

COMMIT=94b997186329768cc91ca6d889215627b513b802
OUT=${1:?usage: run.sh <output dir> [num_sizes num_per_size allow_move]}
RULES=("${2:-3}" "${3:-2}" "${4:-1}")

mkdir -p "$OUT"
cd "$OUT"
if [ ! -f gobblet.cpp ]; then
    curl -fsSL "https://raw.githubusercontent.com/possibly-wrong/gobblet/$COMMIT/gobblet.cpp" -o gobblet.orig.cpp
    # The upstream source has CRLF line endings.
    tr -d '\r' < gobblet.orig.cpp | sed 's/^    game\.play();$/    return 0;/' > gobblet.cpp
    grep -q '^    return 0;$' gobblet.cpp
fi
g++ -O3 -march=native -std=c++20 -o gobblet gobblet.cpp

echo "rules: ${RULES[*]}"
/usr/bin/time -v ./gobblet <<< "${RULES[*]}" 2>&1
ls -l gobblet_*.dat
