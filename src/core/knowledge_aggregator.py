"""Cross-session knowledge aggregation.

Reads all saved session journals, clusters error patterns,
auto-promotes patterns seen in multiple sessions.

Usage:
    from core.knowledge_aggregator import aggregate_sessions

    promoted = aggregate_sessions(Path("data/sessions"), min_sessions=3)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from core.session_journal import SessionJournal
from core.session_analyzer import CandidateKnowledge, analyze_journal


@dataclass
class AggregatedPattern:
    """A pattern seen across multiple sessions — high-confidence knowledge."""
    title: str
    category: str
    solver: str
    physics: str
    description: str
    session_count: int
    confidence: float
    session_ids: list[str] = field(default_factory=list)
    fix_diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "solver": self.solver,
            "physics": self.physics,
            "description": self.description,
            "session_count": self.session_count,
            "confidence": round(self.confidence, 2),
            "session_ids": self.session_ids,
            "fix_diff": self.fix_diff,
        }


def normalize_error(msg: str) -> str:
    """Normalize error messages for structural comparison.

    Replaces numbers, file paths, and UUIDs with tokens so that
    "expected 4 got 3" and "expected 8 got 7" match as the same pattern.
    """
    msg = re.sub(r'\b\d+(\.\d+)?\b', 'NUM', msg)
    msg = re.sub(r'/[^\s:,]+', 'PATH', msg)
    msg = re.sub(r'[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', 'UUID', msg)
    msg = re.sub(r'0x[0-9a-f]+', 'ADDR', msg)
    return msg.lower().strip()


def aggregate_sessions(
    session_dir: Path,
    min_sessions: int = 2,
) -> list[AggregatedPattern]:
    """Load all session journals, cluster error patterns, promote common ones.

    Args:
        session_dir: Directory containing session_*.json files.
        min_sessions: Minimum number of different sessions a pattern must
                      appear in to be promoted (default 2).

    Returns:
        List of promoted patterns, sorted by session count (highest first).
    """
    if not session_dir.exists():
        return []

    # Collect all candidates from all sessions
    all_candidates: list[tuple[str, CandidateKnowledge]] = []  # (session_id, candidate)
    for session_file in sorted(session_dir.glob("session_*.json")):
        try:
            journal = SessionJournal.load(session_file)
            candidates = analyze_journal(journal)
            for c in candidates:
                all_candidates.append((journal.session_id, c))
        except Exception:
            continue

    if not all_candidates:
        return []

    # Cluster by (solver, tool_name in title) + normalized error similarity
    clusters = _cluster_candidates(all_candidates)

    # Promote clusters seen in >= min_sessions different sessions
    promoted = []
    for cluster in clusters:
        unique_sessions = set(sid for sid, _ in cluster)
        if len(unique_sessions) < min_sessions:
            continue

        # Pick the best candidate from the cluster
        best = max(cluster, key=lambda x: x[1].confidence)
        _, best_candidate = best

        # Boost confidence based on cross-session validation
        boosted = min(0.95, best_candidate.confidence + 0.05 * len(unique_sessions))

        promoted.append(AggregatedPattern(
            title=best_candidate.title,
            category=best_candidate.category,
            solver=best_candidate.solver,
            physics=best_candidate.physics,
            description=best_candidate.description,
            session_count=len(unique_sessions),
            confidence=boosted,
            session_ids=sorted(unique_sessions),
            fix_diff=best_candidate.fix_diff,
        ))

    promoted.sort(key=lambda p: p.session_count, reverse=True)
    return promoted


def save_aggregated(patterns: list[AggregatedPattern], output_path: Path):
    """Save aggregated patterns to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "patterns": [p.to_dict() for p in patterns],
    }
    output_path.write_text(json.dumps(data, indent=2))


def _cluster_candidates(
    candidates: list[tuple[str, CandidateKnowledge]],
    similarity_threshold: float = 0.6,
) -> list[list[tuple[str, CandidateKnowledge]]]:
    """Cluster candidates by structural similarity of their titles.

    Groups by (solver, category) first, then by normalized title similarity.
    """
    # Group by (solver, category) for cheap initial bucketing
    groups: dict[tuple, list[tuple[str, CandidateKnowledge]]] = defaultdict(list)
    for sid, c in candidates:
        key = (c.solver, c.category)
        groups[key].append((sid, c))

    all_clusters = []
    for key, group in groups.items():
        # Within each group, cluster by normalized title similarity
        clusters: list[list[tuple[str, CandidateKnowledge]]] = []
        for item in group:
            sid, c = item
            norm_title = normalize_error(c.title)
            placed = False
            for cluster in clusters:
                # Compare against first item in cluster
                ref_title = normalize_error(cluster[0][1].title)
                ratio = SequenceMatcher(None, norm_title, ref_title).ratio()
                if ratio > similarity_threshold:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
        all_clusters.extend(clusters)

    return all_clusters
