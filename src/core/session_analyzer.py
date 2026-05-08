"""Post-session analyzer — extracts candidate knowledge from session journals.

Reads a session journal (in-memory or from disk) and detects patterns
that indicate the agent learned something potentially useful for future
sessions. Candidates are scored by confidence and checked for novelty
against the existing knowledge base.

Patterns detected:
  1. Error→success: tool failed, agent retried, succeeded
  2. Knowledge-after-failure: tool failed, agent consulted knowledge
  3. Source reading: agent browsed solver source (documentation gap)
  4. Convergence issues: coupling needed many iterations or non-default relaxation
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from core.session_journal import SessionJournal


@dataclass
class CandidateKnowledge:
    """A potential knowledge contribution extracted from a session."""

    category: str       # "pitfall", "workaround", "doc_gap", "parameter_tip"
    solver: str
    physics: str
    title: str          # one-line summary
    description: str    # detailed explanation
    confidence: float   # 0.0 to 1.0
    fix_diff: str = ""  # what changed between failing and succeeding input
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_journal(journal: SessionJournal) -> list[CandidateKnowledge]:
    """Analyze a session journal and extract candidate knowledge.

    Args:
        journal: A SessionJournal (in-memory or loaded from disk).

    Returns:
        List of CandidateKnowledge entries, sorted by confidence (highest first).
    """
    if not journal.events:
        return []

    events = [_event_dict(e) for e in journal.events]
    candidates: list[CandidateKnowledge] = []

    candidates.extend(_detect_error_then_success(events))
    candidates.extend(_detect_knowledge_after_failure(events))
    candidates.extend(_detect_source_reading(events))
    candidates.extend(_detect_convergence_issues(events))

    # Deduplicate by title similarity
    candidates = _deduplicate(candidates)

    # Sort by confidence (highest first)
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def filter_against_existing(
    candidates: list[CandidateKnowledge],
    existing_pitfalls: list[str],
    threshold: float = 0.5,
) -> list[CandidateKnowledge]:
    """Filter candidates by confidence and novelty against existing knowledge.

    Args:
        candidates: Raw candidates from analyze_journal.
        existing_pitfalls: List of existing pitfall strings to check against.
        threshold: Minimum confidence to keep (default 0.5).

    Returns:
        Filtered list with only novel, high-confidence candidates.
    """
    filtered = []
    for c in candidates:
        if c.confidence < threshold:
            continue
        if _is_known(c, existing_pitfalls):
            continue
        filtered.append(c)
    return filtered


def analyze_journal_file(path: str | Path) -> list[CandidateKnowledge]:
    """Convenience: load a journal from disk and analyze it."""
    journal = SessionJournal.load(path)
    return analyze_journal(journal)


def format_candidates(candidates: list[CandidateKnowledge]) -> str:
    """Format candidates as a human-readable string for the session_insights tool."""
    if not candidates:
        return "No knowledge candidates discovered in this session."

    lines = [f"## Session Insights ({len(candidates)} candidate{'s' if len(candidates) != 1 else ''})\n"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"### [{i}] {c.category.upper()}: {c.title}")
        lines.append(f"Confidence: {c.confidence:.1f} | Solver: {c.solver or 'n/a'} | Physics: {c.physics or 'n/a'}")
        lines.append(c.description)
        if c.fix_diff:
            lines.append(f"**What changed:** {c.fix_diff}")
        lines.append("")
    return "\n".join(lines)


# ── Pattern detectors ────────────────────────────────────────


def _detect_error_then_success(events: list[dict]) -> list[CandidateKnowledge]:
    """Pattern 1: tool fails, agent retries (possibly after fixes), succeeds.

    When input_snapshot is available, computes the diff between failing
    and succeeding inputs — this diff IS the knowledge.
    """
    candidates = []
    for i, evt in enumerate(events):
        if evt.get("event_type") != "tool_error":
            continue
        tool = evt.get("tool_name", "")
        solver = evt.get("solver", "")
        # Look ahead for a success on the same tool+solver
        for j in range(i + 1, min(i + 15, len(events))):
            later = events[j]
            if (later.get("event_type") == "tool_success"
                    and later.get("tool_name") == tool
                    and later.get("solver") == solver):
                steps_between = j - i
                # Higher confidence if fewer steps (direct fix)
                confidence = min(0.9, 0.5 + 0.1 * (5 - steps_between))
                confidence = max(0.3, confidence)
                error_msg = evt.get("error_message", "unknown error")

                # Compute input diff if snapshots available
                fail_snap = evt.get("input_snapshot", {})
                succ_snap = later.get("input_snapshot", {})
                fix_diff = _diff_snapshots(fail_snap, succ_snap)

                desc = f"Tool `{tool}` failed with: {error_msg}."
                if fix_diff:
                    desc += f" Fixed by: {fix_diff}."
                else:
                    desc += f" Resolved after {steps_between} step(s)."

                candidates.append(CandidateKnowledge(
                    category="pitfall",
                    solver=solver,
                    physics=evt.get("physics", ""),
                    title=f"{tool}: {error_msg[:80]}",
                    description=desc,
                    fix_diff=fix_diff,
                    confidence=round(confidence, 2),
                    evidence=[evt, later],
                ))
                break  # only match first success per error
    return candidates


def _detect_knowledge_after_failure(events: list[dict]) -> list[CandidateKnowledge]:
    """Pattern 2: tool fails, then agent looks up knowledge — suggests gap."""
    candidates = []
    for i, evt in enumerate(events):
        if evt.get("event_type") != "tool_error":
            continue
        for j in range(i + 1, min(i + 5, len(events))):
            later = events[j]
            if later.get("event_type") == "knowledge_lookup":
                candidates.append(CandidateKnowledge(
                    category="doc_gap",
                    solver=evt.get("solver", ""),
                    physics=later.get("physics", evt.get("physics", "")),
                    title=f"Knowledge consulted after {evt.get('tool_name', '')} failure",
                    description=(
                        f"Agent failed at `{evt.get('tool_name', '')}` "
                        f"({evt.get('error_message', 'no details')[:100]}), "
                        f"then looked up knowledge "
                        f"({later.get('notes', 'no details')[:100]}). "
                        f"Existing knowledge may be insufficient."
                    ),
                    confidence=0.5,
                    evidence=[evt, later],
                ))
                break
    return candidates


def _detect_source_reading(events: list[dict]) -> list[CandidateKnowledge]:
    """Pattern 3: agent browsed solver source — documentation gap."""
    candidates = []
    for evt in events:
        if evt.get("event_type") != "source_read":
            continue
        candidates.append(CandidateKnowledge(
            category="doc_gap",
            solver=evt.get("solver", ""),
            physics="",
            title=f"Agent read {evt.get('solver', '')} source: {evt.get('notes', '')[:60]}",
            description=(
                f"Agent needed to browse solver source code "
                f"({evt.get('notes', 'no details')[:100]}), "
                f"suggesting the knowledge base is incomplete for this area."
            ),
            confidence=0.4,
            evidence=[evt],
        ))
    return candidates


def _detect_convergence_issues(events: list[dict]) -> list[CandidateKnowledge]:
    """Pattern 4: coupling needed many iterations or non-default parameters."""
    candidates = []
    for evt in events:
        if evt.get("event_type") != "convergence_issue":
            continue
        details = evt.get("details", {})
        candidates.append(CandidateKnowledge(
            category="parameter_tip",
            solver=evt.get("solver", ""),
            physics=evt.get("physics", "coupling"),
            title=(
                f"Coupling required {details.get('iterations', '?')} iterations"
                f" (relaxation={details.get('relaxation', '?')})"
            ),
            description=(
                f"Non-trivial convergence detected in coupled solve. "
                f"Details: {json.dumps(details)}"
            ),
            confidence=0.6,
            evidence=[evt],
        ))
    return candidates


def _diff_snapshots(fail: dict, succ: dict) -> str:
    """Compute human-readable diff between failing and succeeding input snapshots.

    Returns a description of what changed, or empty string if no snapshots.
    """
    if not fail and not succ:
        return ""
    if not fail or not succ:
        return ""

    diffs = []
    all_keys = sorted(set(fail) | set(succ))
    for k in all_keys:
        fv = fail.get(k, "<absent>")
        sv = succ.get(k, "<absent>")
        if fv != sv:
            diffs.append(f"{k}: {fv!r} -> {sv!r}")

    if not diffs:
        return "inputs structurally identical (fix was in content)"
    return "; ".join(diffs)


# ── Helpers ──────────────────────────────────────────────────


def _event_dict(evt) -> dict:
    """Convert a JournalEvent to a plain dict for pattern matching."""
    return {
        "event_type": evt.event_type,
        "tool_name": evt.tool_name,
        "solver": evt.solver,
        "physics": evt.physics,
        "details": evt.details,
        "error_message": evt.error_message,
        "notes": evt.notes,
        "timestamp": evt.timestamp,
        "input_snapshot": evt.input_snapshot,
    }


def _is_known(candidate: CandidateKnowledge, existing: list[str]) -> bool:
    """Check if a candidate is already covered by existing knowledge."""
    for text in existing:
        # Fuzzy match: if title or description overlaps > 60% with existing
        ratio = SequenceMatcher(
            None, candidate.title.lower(), text.lower()
        ).ratio()
        if ratio > 0.6:
            return True
    return False


def _deduplicate(candidates: list[CandidateKnowledge]) -> list[CandidateKnowledge]:
    """Remove near-duplicate candidates (same title similarity > 80%).

    When duplicates are found, keeps the higher-confidence one.
    """
    unique: list[CandidateKnowledge] = []
    for c in candidates:
        dup_index = -1
        for i, existing in enumerate(unique):
            ratio = SequenceMatcher(
                None, c.title.lower(), existing.title.lower()
            ).ratio()
            if ratio > 0.8:
                dup_index = i
                break
        if dup_index >= 0:
            # Replace with higher-confidence version
            if c.confidence > unique[dup_index].confidence:
                unique[dup_index] = c
        else:
            unique.append(c)
    return unique
