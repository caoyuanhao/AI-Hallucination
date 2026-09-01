"""
Unit tests for the hallucination reducer (no API calls required).
Tests RAG retrieval and prompt building independently.
"""
import unittest
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.reducer.rag_system import RAGSystem, Document
from src.reducer.prompt_strategies import build_system_prompt
from src.reducer.hallucination_reducer import HallucinationReducer


class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        self.rag = RAGSystem()
        self.rag.add_texts([
            "Alexander Graham Bell patented the telephone in 1876.",
            "The Eiffel Tower was completed in 1889.",
            "Einstein received the Nobel Prize in Physics in 1921.",
        ])

    def test_retrieve_relevant_document(self):
        result = self.rag.retrieve("telephone patent Bell", top_k=1)
        self.assertTrue(len(result.documents) > 0)
        self.assertIn("1876", result.documents[0].text)

    def test_retrieve_returns_scores(self):
        result = self.rag.retrieve("Eiffel Tower construction")
        self.assertTrue(len(result.scores) > 0)
        self.assertGreater(result.scores[0], 0)

    def test_context_text_formatted(self):
        result = self.rag.retrieve("Nobel Prize Einstein")
        self.assertIn("[Source", result.context_text)

    def test_empty_rag_returns_empty(self):
        rag = RAGSystem()
        result = rag.retrieve("anything")
        self.assertEqual(result.documents, [])
        self.assertEqual(result.context_text, "")

    def test_add_documents_with_source(self):
        rag = RAGSystem()
        rag.add_documents([Document("Paris is the capital of France.", source="geography")])
        result = rag.retrieve("capital of France")
        self.assertGreater(len(result.documents), 0)
        self.assertEqual(result.documents[0].source, "geography")

    def test_long_document_gets_chunked(self):
        rag = RAGSystem(chunk_size=10, chunk_overlap=2)
        long_text = " ".join([f"word{i}" for i in range(50)])
        rag.add_texts([long_text])
        self.assertGreater(len(rag._chunks), 1)

    def test_clear_resets_store(self):
        self.rag.clear()
        result = self.rag.retrieve("telephone")
        self.assertEqual(result.documents, [])


class TestPromptStrategies(unittest.TestCase):

    def test_base_prompt_contains_rules(self):
        prompt = build_system_prompt()
        self.assertIn("ACCURACY FIRST", prompt)
        self.assertIn("SAY I DON'T KNOW", prompt)

    def test_cot_addition(self):
        prompt_with = build_system_prompt(use_chain_of_thought=True)
        prompt_without = build_system_prompt(use_chain_of_thought=False)
        self.assertIn("reason through", prompt_with)
        self.assertNotIn("reason through", prompt_without)

    def test_context_injected(self):
        prompt = build_system_prompt(context="Paris is the capital of France.")
        self.assertIn("Paris is the capital", prompt)
        self.assertIn("RETRIEVED CONTEXT", prompt)


class TestHallucinationReducer(unittest.TestCase):

    def _make_reducer(self, response_text: str = "Corrected answer.") -> HallucinationReducer:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=response_text)]
        )
        return HallucinationReducer(client=mock_client)

    def test_reduce_returns_result(self):
        reducer = self._make_reducer("Bell patented the telephone in 1876.")
        result = reducer.reduce(
            query="When did Bell patent the telephone?",
            original_response="Bell did it in 1885.",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.reduced_response, "Bell patented the telephone in 1876.")

    def test_rag_context_retrieved_when_knowledge_added(self):
        reducer = self._make_reducer()
        reducer.add_knowledge(["Bell patented the telephone in 1876."])
        result = reducer.reduce(query="When did Bell patent the telephone?")
        self.assertIsNotNone(result.retrieval)

    def test_no_rag_when_empty_store(self):
        reducer = self._make_reducer()
        result = reducer.reduce(query="What is 2+2?")
        # retrieval should be None or have empty docs since store is empty
        if result.retrieval is not None:
            self.assertEqual(result.retrieval.documents, [])

    def test_strategy_string_populated(self):
        reducer = self._make_reducer()
        result = reducer.reduce(query="test")
        self.assertTrue(len(result.strategy_used) > 0)


if __name__ == "__main__":
    unittest.main()
