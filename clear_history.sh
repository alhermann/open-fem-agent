#!/bin/bash
# Clear conversation history before launching fresh Claude instance
rm -rf /home/alexander/.claude/projects/-home-alexander-Schreibtisch-open-fem-agent/*.jsonl
rm -rf /home/alexander/.claude/projects/-home-alexander-Schreibtisch-open-fem-agent/*/
echo "History cleared."
