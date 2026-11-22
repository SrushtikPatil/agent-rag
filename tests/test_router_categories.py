import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from core.agentic_router import AgenticAssistant, AgentState

class TestRouterCategories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assistant = AgenticAssistant()

    def test_general_question_routing(self):
        # General knowledge questions should NOT be trapped in RAG or link analysis
        route1 = self.assistant.classify_intent("Explain quantum computing simply", file_type=None, active_doc_id=None)
        self.assertEqual(route1, "GENERAL_CHAT")

        route2 = self.assistant.classify_intent("Write a python function to check if a number is prime", file_type=None, active_doc_id=None)
        self.assertEqual(route2, "GENERAL_CHAT")

        route3 = self.assistant.classify_intent("What is the capital of France?", file_type=None, active_doc_id=None)
        self.assertEqual(route3, "GENERAL_CHAT")

    def test_document_question_routing(self):
        # Questions asking specifically about documents or matching indexed docs with high confidence
        route = self.assistant.classify_intent("What is the CIA triad in database security?", file_type=None, active_doc_id=None)
        self.assertEqual(route, "MULTIMODAL_RAG")

        # Questions referring explicitly to documents
        route2 = self.assistant.classify_intent("Summarize the document", file_type=None, active_doc_id=None)
        self.assertEqual(route2, "MULTIMODAL_RAG")

    def test_link_routing(self):
        # Links should route to LINK_ANALYSIS
        route = self.assistant.classify_intent("https://youtu.be/9-XJfZnXh3Q", file_type=None, active_doc_id=None)
        self.assertEqual(route, "LINK_ANALYSIS")

        route2 = self.assistant.classify_intent("Analyze this link https://example.com/article", file_type=None, active_doc_id=None)
        self.assertEqual(route2, "LINK_ANALYSIS")

    def test_news_routing(self):
        # Live news queries should route to LIVE_NEWS
        route = self.assistant.classify_intent("What is today's latest breaking news?", file_type=None, active_doc_id=None)
        self.assertEqual(route, "LIVE_NEWS")

if __name__ == "__main__":
    unittest.main()
