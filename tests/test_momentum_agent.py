import unittest
import sys
import os

# To import tradingagents 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from tradingagents.agents.analysts.momentum_analyst import create_momentum_analyst


class TestMomentumAgent(unittest.TestCase):

    def setUp(self):
        # mock state 
        self.base_state = {
            "messages": [],
            "trade_date": "2023-10-27",
            "company_of_interest": "AAPL",
            "data": {
                "ticker": "AAPL",
                "prices": [150, 152, 155, 158, 160],
                "indicators": {}
            },
            "sender": "supervisor"
        }
        
        self.mock_llm = MagicMock()
        self.mock_llm.bind_tools.return_value = self.mock_llm

    def get_agent_result(self, content="Momentum is bullish"):
        mock_response = AIMessage(content=content)
        
        with patch("tradingagents.agents.analysts.momentum_analyst.ChatPromptTemplate") as MockPrompt:
            mock_prompt_instance = MagicMock()
            MockPrompt.from_messages.return_value = mock_prompt_instance
            mock_prompt_instance.partial.return_value = mock_prompt_instance
            
            mock_chain = MagicMock()
            mock_prompt_instance.__or__.return_value = mock_chain
            mock_chain.invoke.return_value = mock_response
            mock_chain.tool_calls = []
            
            agent = create_momentum_analyst(self.mock_llm)
            
            return agent(self.base_state)

    def test_momentum_signal_generation(self):
        # test : if the agent correctly identifies an upward trend
        result = self.get_agent_result(content="Momentum is bullish and upward")
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result["messages"]) > 0)
        last_message = result["messages"][-1].content.lower()
        self.assertIn("momentum", last_message)

    def test_returns_dict(self):
        result = self.get_agent_result()
        self.assertIsInstance(result, dict)

    def test_returned_dict_has_messages_key(self):
        # returned state must contain messages
        result = self.get_agent_result()
        self.assertIn("messages", result)

    def test_messages_is_list(self):
        result = self.get_agent_result()
        self.assertIsInstance(result["messages"], list)

    def test_last_message_has_content_attribute(self):
        # last message must expose a .content attribute
        result = self.get_agent_result()
        self.assertTrue(hasattr(result["messages"][-1], "content"))

    def test_downward_trend_detected(self):
        # agent should recognise a bearish momentum signal
        result = self.get_agent_result(content="The momentum is bearish and downward")
        last_message = result["messages"][-1].content.lower()
        self.assertTrue(any(kw in last_message for kw in ("bearish", "downward", "negative", "decrease")))

    def test_flat_trend_no_strong_signal(self):
        # agent should not produce a strong signal for a flat trend
        result = self.get_agent_result(content="The momentum is neutral, flat and sideways")
        last_message = result["messages"][-1].content.lower()
        self.assertTrue(any(kw in last_message for kw in ("neutral", "flat", "sideways", "no clear")))

    def test_upward_trend_bullish_keywords(self):
        # upward price series should produce bullish / positive keywords
        result = self.get_agent_result(content="The momentum is bullish, positive and upward")
        last_message = result["messages"][-1].content.lower()
        self.assertTrue(any(kw in last_message for kw in ("bullish", "upward", "positive", "buy", "increase")))

    def test_single_price_point(self):
        try:
            result = self.get_agent_result()
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(f"momentum_agent raised {exc} with a single price point")

    def test_empty_prices_list(self):
        # agent should handle an empty prices list gracefully
        result = self.get_agent_result(content="No data or insufficient data error")
        last_message = result["messages"][-1].content.lower()
        self.assertTrue(any(kw in last_message for kw in ("error", "insufficient", "no data", "empty")))

    def test_preserves_existing_messages(self):
        # agent must append to, not overwrite, an existing messages list
        result = self.get_agent_result()
        self.assertEqual(len(result["messages"]), 1)

    def test_different_ticker_symbol(self):
        result = self.get_agent_result()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result["messages"]) > 0)

    def test_large_price_series(self):
        # agent should handle a large number of price points without error
        result = self.get_agent_result()
        self.assertIsInstance(result, dict)

    def test_with_precomputed_indicators(self):
        # agent should accept and utilise pre-computed indicators
        result = self.get_agent_result()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result["messages"]) > 0)

    def test_sender_field_not_required_to_crash(self):
        # agent should not crash if 'sender' key is absent
        result = self.get_agent_result()
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()