#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-.}"

mkdir -p "$DEST/.claude/skills"
cp -R "$SRC_DIR/.claude/skills/." "$DEST/.claude/skills/"

echo "Installed Claude Code skills into: $DEST/.claude/skills"
echo "Available skills:"
find "$DEST/.claude/skills" -mindepth 2 -maxdepth 2 -name SKILL.md -print | sed 's#^.*/skills/#/#' | sed 's#/SKILL.md##'
