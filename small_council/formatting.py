from __future__ import annotations


def render_fallback(
    winner: str | None,
    why: str,
    votes: list[dict],
    leaderboard: list[dict],
    tied_options: list[str] | None = None,
    winning_members: list[str] | None = None,
    tie_broken_by: str | None = None,
) -> str:
    decision_lines = tied_options if tied_options else [winner or "No decision reached."]
    if not tied_options and winning_members and len(winning_members) > 1:
        decision_lines.append(f"Winning proposers: {', '.join(winning_members)}")
    if not tied_options and tie_broken_by:
        decision_lines.append(f"Tie broken by: {tie_broken_by}")
    reason_heading = "Why It Tied" if tied_options else "Why It Won"
    lines = [
        "==================================================",
        "Decision",
        "==================================================",
        "",
        *decision_lines,
        "",
        "==================================================",
        reason_heading,
        "==================================================",
        "",
        why,
        "",
        "==================================================",
        "Council Votes",
        "==================================================",
        "",
    ]
    for vote in votes:
        lines.append(f"{vote['voter']} -> {vote['selected_option']}")
        lines.append(f"Reason: \"{vote['reason']}\"")
        lines.append("")
    lines.extend(
        [
            "==================================================",
            "Leaderboard",
            "==================================================",
            "",
        ]
    )
    for index, row in enumerate(leaderboard, start=1):
        crown = " President" if row["president"] else ""
        lines.append(f"{index}. {row['member']} - {row['total_wins']} wins{crown}")
    return "\n".join(lines).rstrip() + "\n"
