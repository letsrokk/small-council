from __future__ import annotations

import unittest

from small_council.decision import (
    canonical_recommendations,
    decision_from_rounds,
    evaluate_vote_round,
    validate_recommendation_groups,
)


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


class RecommendationGroupingTests(unittest.TestCase):
    def test_groups_pizza_and_italian_flatbread_aliases(self) -> None:
        recommendations = [
            {"proposer": "Aurelia", "recommendation": "Pizza"},
            {"proposer": "Echo", "recommendation": "Italian flatbread with cheese and tomato sauce"},
            {"proposer": "Bram", "recommendation": "Sushi"},
        ]
        groups = validate_recommendation_groups(
            [
                {
                    "canonical_option": "Pizza",
                    "proposers": ["Aurelia", "Echo"],
                    "member_recommendations": [
                        "Pizza",
                        "Italian flatbread with cheese and tomato sauce",
                    ],
                    "reason": "Same final food option.",
                },
                {
                    "canonical_option": "Sushi",
                    "proposers": ["Bram"],
                    "member_recommendations": ["Sushi"],
                    "reason": "Distinct option.",
                },
            ],
            recommendations,
        )

        self.assertEqual(
            [
                {
                    "canonical_option": "Pizza",
                    "proposers": ["Aurelia", "Echo"],
                    "member_recommendations": [
                        "Pizza",
                        "Italian flatbread with cheese and tomato sauce",
                    ],
                    "reason": "Same final food option.",
                },
                {
                    "canonical_option": "Sushi",
                    "proposers": ["Bram"],
                    "member_recommendations": ["Sushi"],
                    "reason": "Distinct option.",
                },
            ],
            groups,
        )

    def test_splits_incoherent_group_with_pizza_and_sushi(self) -> None:
        recommendations = [
            {"proposer": "Aurelia", "recommendation": "Pizza"},
            {"proposer": "Bram", "recommendation": "Sushi"},
            {"proposer": "Cato", "recommendation": "Sushi"},
            {"proposer": "Echo", "recommendation": "Pizza"},
        ]
        groups = validate_recommendation_groups(
            [
                {
                    "canonical_option": "Sushi",
                    "proposers": ["Aurelia", "Bram", "Cato", "Echo"],
                    "member_recommendations": ["Pizza", "Sushi", "Sushi", "Pizza"],
                    "reason": "All recommendations converge on Sushi.",
                }
            ],
            recommendations,
        )

        self.assertEqual(["Pizza", "Sushi"], [group["canonical_option"] for group in groups])
        self.assertEqual(["Aurelia", "Echo"], groups[0]["proposers"])
        self.assertEqual(["Bram", "Cato"], groups[1]["proposers"])

    def test_canonical_recommendations_keep_split_groups_distinct(self) -> None:
        recommendations = [
            {"proposer": "Aurelia", "recommendation": "Pizza"},
            {"proposer": "Bram", "recommendation": "Sushi"},
        ]
        groups = validate_recommendation_groups(
            [
                {
                    "canonical_option": "Sushi",
                    "proposers": ["Aurelia", "Bram"],
                    "member_recommendations": ["Pizza", "Sushi"],
                    "reason": "Bad merge.",
                }
            ],
            recommendations,
        )

        voting_options = [item["recommendation"] for item in canonical_recommendations(groups)]

        self.assertEqual(["Pizza", "Sushi"], voting_options)


if __name__ == "__main__":
    unittest.main()
