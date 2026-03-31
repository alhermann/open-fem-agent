#!/bin/bash
# Clear conversation history before launching a fresh agent session.
# This prevents anchoring bias from previous conversations.

# Derive the project-specific Claude history directory from the current path
PROJECT_DIR=$(pwd)
# Claude Code encodes the project path with dashes replacing slashes
ENCODED_PATH=$(echo "$PROJECT_DIR" | sed 's|^/||;s|/|-|g')
HISTORY_DIR="$HOME/.claude/projects/-$ENCODED_PATH"

if [ -d "$HISTORY_DIR" ]; then
    rm -rf "$HISTORY_DIR"/*.jsonl
    rm -rf "$HISTORY_DIR"/*/
    echo "History cleared: $HISTORY_DIR"
else
    echo "No history directory found at: $HISTORY_DIR"
    echo "Run this script from the open-fem-agent project root."
fi
