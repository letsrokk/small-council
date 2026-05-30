from __future__ import annotations

import unittest

from small_council.decision import decision_from_rounds, evaluate_vote_round


class DecisionTieBreakTests(unittest.TestCase):
    def test_president_tie_break_resolves_final_tie(self) -> None:
        recommendations = [
            {"proposer": "Aurelia", "proposers": ["Aurelia"], "recommendation": "Option A"},
            {"proposer": "Bram", "proposers": ["Bram"], "recommendation": "Option B"},
        ]
        round_result = evaluate_vote_round(
            recommendations,
            [
                {"voter": "Aurelia", "selected_option": "Option A"},
                {"voter": "Bram", "selected_option": "Option B"},
            ],
            1,
        )
        tie_break_vote = {
            "voter": "Bram",
            "selected_option": "Option B",
            "reason": "President's call.",
            "tie_break": True,
        }

        decision = decision_from_rounds(
            recommendations,
            [round_result],
            tie_breaker_member="Bram",
            tie_break_vote=tie_break_vote,
        )

        self.assertEqual("resolved", decision.status)
        self.assertEqual("Option B", decision.winning_option)
        self.assertEqual("Bram", decision.winning_member)
        self.assertEqual(["Bram"], decision.winning_members)
        self.assertEqual("Bram", decision.tie_broken_by)
        self.assertEqual(tie_break_vote, decision.tie_break_vote)
        self.assertEqual([], decision.tied_options)

    def test_invalid_president_tie_break_preserves_unresolved_tie(self) -> None:
        recommendations = [
            {"proposer": "Aurelia", "proposers": ["Aurelia"], "recommendation": "Option A"},
            {"proposer": "Bram", "proposers": ["Bram"], "recommendation": "Option B"},
        ]
        round_result = evaluate_vote_round(
            recommendations,
            [
                {"voter": "Aurelia", "selected_option": "Option A"},
                {"voter": "Bram", "selected_option": "Option B"},
            ],
            1,
        )

        decision = decision_from_rounds(
            recommendations,
            [round_result],
            tie_breaker_member="Bram",
            tie_break_vote={"voter": "Bram", "selected_option": "Option C"},
        )

        self.assertEqual("unresolved_tie", decision.status)
        self.assertIsNone(decision.winning_option)
        self.assertIsNone(decision.tie_broken_by)
        self.assertIsNone(decision.tie_break_vote)
        self.assertEqual(["Option A", "Option B"], decision.tied_options)


if __name__ == "__main__":
    unittest.main()
