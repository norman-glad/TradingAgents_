import unittest
from unittest.mock import MagicMock, patch
# Import your actual momentum agent function here
from tradingagents.agents.analysts import momentum_agent 

class TestMomentumAgent(unittest.TestCase):

    def setUp(self):
        """Set up a dummy state that the agent expects."""
        self.base_state = {
            "messages": [],
            "data": {
                "ticker": "AAPL",
                "prices": [150, 152, 155, 158, 160], # Upward trend
                "indicators": {}
            },
            "sender": "supervisor"
        }

    def test_momentum_signal_generation(self):
        """Test if the agent correctly identifies an upward trend."""
        # Execute the agent logic
        result = momentum_agent(self.base_state)
        
        # Check if the result is a dictionary (LangGraph state)
        self.assertIsInstance(result, dict)
        # Check if 'messages' was updated
        self.assertTrue(len(result["messages"]) > 0)
        # Check for specific momentum keywords in the output
        last_message = result["messages"][-1].content.lower()
        self.assertIn("momentum", last_message)

    @patch('tradingagents.agents.analysts.get_financial_data') # Mock the data fetching tool
    def test_agent_with_empty_data(self, mock_get_data):
        """Test how the agent handles a scenario where no price data is found."""
        mock_get_data.return_value = None 
        
        # If your agent is robust, it should return an error message rather than crashing
        result = momentum_agent(self.base_state)
        self.assertIn("error", result["messages"][-1].content.lower())

if __name__ == "__main__":
    unittest.main()