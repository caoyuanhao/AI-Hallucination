"""
Unit tests for the hallucination detector (no API calls required).
Tests confidence scorer and consistency checker internals independently.
"""
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.detector.confidence_scorer import ConfidenceScorer
from src.detector.consistency_checker import ConsistencyChecker
from src.detector.hallucination_detector import HallucinationDetector


class TestConfidenceScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = ConfidenceScorer()

    def test_low_risk_general_text(self):
        text = "Photosynthesis converts sunlight into glucose using chlorophyll in plant cells."
        result = self.scorer.score(text)
        self.assertEqual(result.risk_level, "low")
        self.assertGreater(result.score, 0.7)

    def test_medium_risk_hedged_text(self):
        text = (
            "I think the capital might be Paris, but I'm not sure. "
            "It probably has a large population. As far as I know it's in France."
        )
        result = self.scorer.score(text)
        self.assertIn(result.risk_level, ("medium", "high"))
        self.assertLess(result.score, 0.8)

    def test_high_risk_overconfident_specific(self):
        text = (
            "Bell definitely invented the telephone in 1885. "
            "Without doubt, he made $1,234,567 in the first year. "
            "Certainly every historian agrees on John Smith's role."
        )
        result = self.scorer.score(text)
        self.assertEqual(result.risk_level, "high")
        self.assertLess(result.score, 0.5)

    def test_markers_found_populated(self):
        text = "I think maybe it could be true, perhaps."
        result = self.scorer.score(text)
        self.assertTrue(len(result.markers_found) > 0)

    def test_specific_claims_counted(self):
        text = "In 2020, the revenue was $5,000,000 — a 45% increase."
        result = self.scorer.score(text)
        self.assertGreater(result.specific_claims, 0)


class TestConsistencyChecker(unittest.TestCase):

    def _make_checker(self, responses: list[str]) -> ConsistencyChecker:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text=r)]) for r in responses
        ]
        return ConsistencyChecker(client=mock_client, n_samples=len(responses))

    def test_identical_responses_low_risk(self):
        text = "The Eiffel Tower is in Paris, France."
        checker = self._make_checker([text, text, text])
        result = checker.check("Where is the Eiffel Tower?")
        self.assertEqual(result.risk_level, "low")
        self.assertGreater(result.mean_similarity, 0.9)

    def test_contradictory_responses_high_risk(self):
        checker = self._make_checker([
            "Einstein won the Nobel Prize in 1921.",
            "Einstein won the Nobel Prize in 1932 for quantum mechanics.",
            "Einstein never won a Nobel Prize; it was Bohr.",
        ])
        result = checker.check("When did Einstein win the Nobel Prize?")
        self.assertIn(result.risk_level, ("medium", "high"))

    def test_samples_stored(self):
        checker = self._make_checker(["Answer A.", "Answer B."])
        result = checker.check("Question?")
        self.assertEqual(len(result.samples), 2)


class TestHallucinationDetector(unittest.TestCase):

    def _make_detector(self, consistency_responses: list[str]) -> HallucinationDetector:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text=r)]) for r in consistency_responses
        ]
        return HallucinationDetector(
            client=mock_client,
            run_consistency_check=True,
            consistency_samples=len(consistency_responses),
        )

    def test_detect_returns_result(self):
        detector = self._make_detector(["Paris is the capital.", "Paris is the capital.", "Paris."])
        result = detector.detect(
            query="What is the capital of France?",
            response="Paris is definitely the capital of France.",
        )
        self.assertIn(result.overall_risk, ("low", "medium", "high"))
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.consistency)

    def test_no_consistency_check(self):
        mock_client = MagicMock()
        detector = HallucinationDetector(client=mock_client, run_consistency_check=False)
        result = detector.detect(
            query="What is 2+2?",
            response="Two plus two equals four.",
        )
        self.assertIsNone(result.consistency)
        self.assertIsNotNone(result.confidence)

    def test_recommendations_populated_for_high_risk(self):
        detector = self._make_detector([
            "Bell invented telephone in 1880.",
            "Bell invented telephone in 1886.",
            "It was actually Edison who made the phone.",
        ])
        result = detector.detect(
            query="When did Bell invent the telephone?",
            response="Bell definitely invented the telephone in 1880. Certainly no one disputes this.",
        )
        if result.overall_risk == "high":
            self.assertTrue(len(result.recommendations) > 0)


if __name__ == "__main__":
    unittest.main()
