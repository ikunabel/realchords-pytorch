#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE="yh522379@copy23-1.hpc.itc.rwth-aachen.de"
REMOTE_REPO="/home/yh522379/realchords-pytorch"
SSH_SOCKET="/tmp/rsync-hpc-${USER}-$$"

cleanup() {
  ssh -o ControlPath="$SSH_SOCKET" -O exit "$REMOTE" 2>/dev/null || true
}
trap cleanup EXIT

# One password + 2FA for the whole script.
ssh -o ControlMaster=yes -o ControlPath="$SSH_SOCKET" -o ControlPersist=10m -fN "$REMOTE"
RSYNC_RSH="ssh -o ControlPath=$SSH_SOCKET -o ControlMaster=no"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/data/cache/" \
  "$REMOTE:$REMOTE_REPO/data/cache/"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/data/voicings/" \
  "$REMOTE:$REMOTE_REPO/data/voicings/"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/journal/" \
  "$REMOTE:$REMOTE_REPO/journal/"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/logs/eval/" \
  "$REMOTE:$REMOTE_REPO/logs/eval/"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/logs/paired_eval/" \
  "$REMOTE:$REMOTE_REPO/logs/paired_eval/"

rsync -avz --progress -e "$RSYNC_RSH" \
  "$ROOT/logs/generated/" \
  "$REMOTE:$REMOTE_REPO/logs/generated/"

echo "Done."
