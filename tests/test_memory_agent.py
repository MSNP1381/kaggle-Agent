import unittest
import os
import json
from states.memory import MemoryAgent
from langchain_openai import ChatOpenAI

class TestMemoryAgent(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_long_term_memory.json"
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.memory_agent = MemoryAgent(llm=self.llm, long_term_file=self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_to_short_term_memory(self):
        self.memory_agent.add_to_short_term_memory("Test content")
        self.assertEqual(len(self.memory_agent.short_term_memory), 1)
        self.assertEqual(self.memory_agent.short_term_memory[0], "Test content")

    def test_add_to_long_term_memory(self):
        test_content = {"type": "test", "content": "Test long-term content"}
        self.memory_agent.add_to_long_term_memory(test_content)
        self.assertEqual(len(self.memory_agent.long_term_memory), 1)
        self.assertEqual(self.memory_agent.long_term_memory[0], test_content)

    def test_get_relevant_context(self):
        self.memory_agent.add_to_short_term_memory("Short-term test content")
        self.memory_agent.add_to_long_term_memory({"type": "test", "content": "Long-term test content"})
        relevant_context = self.memory_agent.get_relevant_context("test content")
        self.assertGreater(len(relevant_context), 0)

    def test_save_and_load_long_term_memory(self):
        test_content = {"type": "test", "content": "Test save and load"}
        self.memory_agent.add_to_long_term_memory(test_content)
        self.memory_agent.save_long_term_memory()

        new_memory_agent = MemoryAgent(llm=self.llm, long_term_file=self.test_file)
        self.assertEqual(len(new_memory_agent.long_term_memory), 1)
        self.assertEqual(new_memory_agent.long_term_memory[0], test_content)

    def test_add_to_conversation(self):
        self.memory_agent.add_to_conversation("user", "Hello")
        self.memory_agent.add_to_conversation("ai", "Hi there!")
        conversation_history = self.memory_agent.get_conversation_history()
        self.assertEqual(len(conversation_history), 2)
        self.assertEqual(conversation_history[0], "user: Hello")
        self.assertEqual(conversation_history[1], "ai: Hi there!")

    def test_clear_memories(self):
        self.memory_agent.add_to_short_term_memory("Short-term test")
        self.memory_agent.add_to_long_term_memory({"type": "test", "content": "Long-term test"})
        
        self.memory_agent.clear_short_term_memory()
        self.assertEqual(len(self.memory_agent.short_term_memory), 0)
        
        self.memory_agent.clear_long_term_memory()
        self.assertEqual(len(self.memory_agent.long_term_memory), 0)

    def test_get_memory_summary(self):
        self.memory_agent.add_to_short_term_memory("Short-term test")
        self.memory_agent.add_to_long_term_memory({"type": "test", "content": "Long-term test"})
        
        summary = self.memory_agent.get_memory_summary()
        self.assertEqual(summary["short_term_memory_size"], 1)
        self.assertEqual(summary["long_term_memory_size"], 1)
        self.assertEqual(summary["long_term_file"], self.test_file)

if __name__ == '__main__':
    unittest.main()
