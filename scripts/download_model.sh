#!/usr/bin/env bash
# Downloads the model a run starts from to $WORK_DIR/models/<name>: the repository at the revision the model's file
# names, from Hugging Face (default) or ModelScope, with a record of what was fetched; a directory matching its record is
# left as it is. On any machine; the installer's target libero_run runs it too.
#   WORK_DIR=<work dir> bash scripts/download_model.sh
#   MODEL_FILE=<a file of examples/models>: the model to download (Qwen3.8-27B's)
#   MODEL_SOURCE=modelscope: use the model's pinned snapshot on modelscope.cn
set -euo pipefail

# the work directory, uv and the Hub's client are the installer's
# shellcheck source=install/common.sh
source "$(dirname "$0")/install/common.sh"

# Read the pins without sourcing the recipe, which requires GPU and training settings.
MODEL_FILE=${MODEL_FILE:-$ROOT/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh}
MODEL_REPO=$(sed -n 's/^MODEL_REPO=//p' "$MODEL_FILE")
MODEL_REVISION=$(sed -n 's/^MODEL_REVISION=//p' "$MODEL_FILE")
MODELSCOPE_REVISION=$(sed -n 's/^MODELSCOPE_REVISION=//p' "$MODEL_FILE")
MODEL_SOURCE=${MODEL_SOURCE:-huggingface}
MODEL_DIR=$WORK_DIR/models/${MODEL_REPO##*/}
# the directory's record of what was fetched
RECORD_NAME=.pear_model

# What the model directory holds, as its record states it: the repository, the revision and every file with its size
# (without the Hub client's own notes under .cache).
model_state() {
  echo "repository $MODEL_REPO"
  echo "revision $MODEL_REVISION"
  if [ "$2" = modelscope ]; then
    echo "modelscope_revision $MODELSCOPE_REVISION"
  fi
  (cd "$1" && find . -type f ! -path './.cache/*' ! -name "$RECORD_NAME" -exec wc -c {} + |
    sed 's/^ *//' | grep -v ' total$' | sort -k 2)
}

if [ -z "$MODEL_REPO" ] || [ -z "$MODEL_REVISION" ]; then
  echo "$MODEL_FILE names no MODEL_REPO and MODEL_REVISION" >&2
  exit 1
fi
case "$MODEL_SOURCE" in
  huggingface) ;;
  modelscope)
    if ! [[ "$MODELSCOPE_REVISION" =~ ^[[:xdigit:]]{40}$ ]]; then
      echo "$MODEL_FILE needs a pinned MODELSCOPE_REVISION to download from ModelScope" >&2
      exit 1
    fi
    ;;
  *) echo "unknown MODEL_SOURCE=$MODEL_SOURCE (huggingface | modelscope)" >&2; exit 1 ;;
esac

# Keep interrupted downloads separate from completed models. Serialize two installers for the same local name.
require_tools flock
DOWNLOADS=$WORK_DIR/models/.downloads/${MODEL_REPO##*/}
mkdir -p "$DOWNLOADS"
exec 9>"$DOWNLOADS/lock"
flock 9
# Recover the old directory if the previous process stopped between the two renames.
if [ ! -e "$MODEL_DIR" ] && [ -d "$DOWNLOADS/previous" ]; then
  mv "$DOWNLOADS/previous" "$MODEL_DIR"
fi
if [ -f "$MODEL_DIR/$RECORD_NAME" ]; then
  recorded_source=huggingface
  if grep -q '^modelscope_revision ' "$MODEL_DIR/$RECORD_NAME"; then
    recorded_source=modelscope
  fi
  if STATE_DIFFERENCE=$(diff "$MODEL_DIR/$RECORD_NAME" <(model_state "$MODEL_DIR" "$recorded_source")); then
    echo "Using the recorded $recorded_source snapshot in $MODEL_DIR"
    exit 0
  fi
  echo "$MODEL_DIR is not what its record states (<: the record, >: the directory) and is fetched anew:"
  echo "$STATE_DIFFERENCE"
fi

require_tools curl getent tar sha256sum
install_uv
snapshot_key=$(printf '%s\n' "$MODEL_SOURCE" "$MODEL_REPO" "$MODEL_REVISION" "$MODELSCOPE_REVISION" | inputs_hash)
staging=$DOWNLOADS/$snapshot_key
mkdir -p "$staging"
echo "Downloading $MODEL_REPO from $MODEL_SOURCE"
if [ "$MODEL_SOURCE" = modelscope ]; then
  MODELSCOPE_CACHE="$WORK_DIR/cache/modelscope" \
    uvx --constraints "$INSTALL_DIR/evalscope/constraints.txt" --from modelscope-hub ms download "$MODEL_REPO" \
    --endpoint https://modelscope.cn --revision "$MODELSCOPE_REVISION" --local-dir "$staging"
else
  download_from_hub model "$MODEL_REPO" "$MODEL_REVISION" "$staging"
fi
model_state "$staging" "$MODEL_SOURCE" > "$staging/$RECORD_NAME"
if [ -e "$MODEL_DIR" ]; then
  if [ -f "$MODEL_DIR/$RECORD_NAME" ]; then
    rm -rf "$DOWNLOADS/previous"
    mv "$MODEL_DIR" "$DOWNLOADS/previous"
  else
    backup=$(mktemp -d "$DOWNLOADS/unrecorded.XXXXXXXX")
    mv "$MODEL_DIR" "$backup/model"
    echo "Previous unrecorded model directory saved at $backup/model"
  fi
fi
mv "$staging" "$MODEL_DIR"
rm -rf "$DOWNLOADS/previous"
