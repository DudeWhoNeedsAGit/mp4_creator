#!/usr/bin/env bash

set -euo pipefail

TARGET_DIR="./mp3"
FORCE=false

# Check for --force flag
if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
fi

# First, list all authors
echo "Current author information:"
for f in "$TARGET_DIR"/*.mp3 "$TARGET_DIR"/*.wav; do
  [ -e "$f" ] || continue
  artist=$(ffprobe -v error -show_entries format_tags=artist -of default=nw=1:nk=1 "$f")
  artist="${artist:-<no artist tag>}"
  echo "$(basename "$f"): $artist"
done

echo
echo "Proceeding to replacement..."

# Replace author where needed
for f in "$TARGET_DIR"/*.mp3 "$TARGET_DIR"/*.wav; do
  [ -e "$f" ] || continue
  artist=$(ffprobe -v error -show_entries format_tags=artist -of default=nw=1:nk=1 "$f")
  artist="${artist:-<no artist tag>}"

  if [[ "$artist" == "Feynman" ]]; then
    echo "Skipping $(basename "$f") — already Feynman"
    continue
  fi

  if $FORCE; then
    answer="y"
  else
    read -rp "Replace artist for $(basename "$f") with 'Feynman'? [y/N]: " answer
  fi

  if [[ "$answer" =~ ^[Yy]$ ]]; then
    tmpfile="$(mktemp --suffix="$(basename "$f")")"
    ffmpeg -i "$f" -metadata artist="Feynman" -codec copy "$tmpfile" -y >/dev/null 2>&1
    mv "$tmpfile" "$f"
    echo "Updated artist for $(basename "$f") to Feynman"
  else
    echo "Skipped $(basename "$f")"
  fi
done
