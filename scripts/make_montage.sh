#!/usr/bin/env bash
# Build a 3x2 montage from three recorded episodes.
#
# Layout -- three columns, best episode on the left:
#
#     front_1     front_2     front_3
#     left_wrist  left_wrist  left_wrist
#
# `front` on top because it shows the whole table, `left_wrist` underneath because it is
# the camera that actually tracks the grasp.  `right_wrist` is deliberately omitted: on
# this two-arm setup the left arm does all the carrying, so the right wrist renders a
# static view that only uses up a third of the frame.
#
# Each input is 256x256 at 20 fps, so a column is 256x512 and the result is 768x512.
# Nothing is scaled or re-encoded for looks -- the frames are the ones the policy saw.
#
# The cut is not cosmetic.  An episode runs to max_steps whether or not the task is
# finished, so the tail after the last cookie lands is dead time.  Pass the cut in seconds;
# the callers use 30 s for 2x5 and 80 s for 2x10, which is where all three columns have
# already reached their final cookie count.
#
# Usage:
#   make_montage.sh --task 2x5 --cut 30 --out PATH DIR1 DIR2 DIR3
set -eu

TASK=""
CUT=""
OUT=""
DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="${2:?--task needs a value}"; shift ;;
        --cut)  CUT="${2:?--cut needs seconds}"; shift ;;
        --out)  OUT="${2:?--out needs a path}"; shift ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) DIRS+=("$1") ;;
    esac
    shift
done

[ -n "$TASK" ] || { echo "--task is required" >&2; exit 2; }
[ -n "$CUT" ]  || { echo "--cut is required" >&2; exit 2; }
[ -n "$OUT" ]  || { echo "--out is required" >&2; exit 2; }
[ "${#DIRS[@]}" -eq 3 ] || { echo "need exactly 3 episode directories, got ${#DIRS[@]}" >&2; exit 2; }

# Refuse early rather than let ffmpeg fail with an unreadable filter graph: a missing input
# is the usual cause and the message it produces points at the graph, not the file.
INPUTS=()
FILTER=""
STACK=""
for i in 0 1 2; do
    dir="${DIRS[$i]}"
    for cam in front left_wrist; do
        f="$dir/$cam.mp4"
        [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
    done
    INPUTS+=(-i "$dir/front.mp4" -i "$dir/left_wrist.mp4")
done

# -ss/-t per input would seek before decoding; here the cut is done in the filter graph so
# all six streams stay locked to the same timeline even if a file's timestamps drift.
for i in 0 1 2; do
    f=$(( i * 2 ))
    w=$(( i * 2 + 1 ))
    FILTER+="[${f}:v]trim=0:${CUT},setpts=PTS-STARTPTS,fps=20,setsar=1[t${i}];"
    FILTER+="[${w}:v]trim=0:${CUT},setpts=PTS-STARTPTS,fps=20,setsar=1[b${i}];"
    FILTER+="[t${i}][b${i}]vstack=inputs=2[c${i}];"
    STACK+="[c${i}]"
done
FILTER+="${STACK}hstack=inputs=3[v]"

mkdir -p "$(dirname "$OUT")"
echo "montage: $TASK  cut=${CUT}s  3 cols x (front/left_wrist)"
for i in 0 1 2; do echo "  col$((i+1)) <- ${DIRS[$i]}"; done

# libx264 crf 18 + yuv420p: the source is 256x256 RGB rendered by MuJoCo, so a low crf
# keeps the cookie edges readable and yuv420p keeps the file playable everywhere.
ffmpeg -hide_banner -loglevel error -y \
    "${INPUTS[@]}" \
    -filter_complex "$FILTER" \
    -map "[v]" -an \
    -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUT"

ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
    -of default=nw=1 "$OUT"
echo "wrote $OUT  ($(du -h "$OUT" | cut -f1))"
