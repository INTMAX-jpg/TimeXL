
import unittest
from unittest.mock import patch, MagicMock
import os
import requests
from training.llm_agents import BaseLLMAgent

class TestLLMAgent(unittest.TestCase):
    
    def setUp(self):
        self.api_key = "test_key_12345"
        self.agent = BaseLLMAgent(api_key=self.api_key)

    @patch('requests.post')
    def test_generate_success(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a generated response."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        prompt = "Hello, world!"
        response = self.agent.generate(prompt)

        # Verify request parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        self.assertEqual(kwargs['headers']['Authorization'], f"Bearer {self.api_key}")
        self.assertEqual(kwargs['json']['messages'][0]['content'], prompt)
        
        # Verify result
        self.assertEqual(response, "This is a generated response.")

    @patch('requests.post')
    def test_generate_api_error(self, mock_post):
        # Mock API error (e.g., 401 Unauthorized)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            self.agent.generate("Test prompt")

    def test_missing_api_key(self):
        # Unset env var if it exists
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
            
        agent = BaseLLMAgent(api_key=None)
        
        with self.assertRaises(ValueError):
            agent.generate("Test prompt")

if __name__ == '__main__':
    unittest.main()
