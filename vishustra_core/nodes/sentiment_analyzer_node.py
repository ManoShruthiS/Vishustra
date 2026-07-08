import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on input text.
    It identifies sentiment as 'positive', 'negative', or 'neutral' based on
    a simplistic keyword matching approach for demonstration purposes.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.

        Expected `data` type: `str` (the text to analyze).

        Expected `context` usage (optional): Could contain configuration for
        sentiment model, language, etc. For this simulation, it's not used
        beyond logging its presence.

        Returns:
            A dictionary containing the original text, the detected sentiment
            ('positive', 'negative', 'neutral'), and a simulated score.

            Example:
            {
                "text": "This product is fantastic!",
                "sentiment": "positive",
                "score": 0.9
            }

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string.
        """
        logger.debug(
            f"[{self.node_name}] Starting sentiment analysis for data type: {type(data).__name__}"
        )
        logger.debug(f"[{self.node_name}] Received context: {context}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            error_msg = f"[{self.node_name}] Input data is an empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_lower = data.lower()
        sentiment = "neutral"
        score = 0.5  # Default neutral score

        # Simple keyword-based sentiment detection for simulation
        positive_keywords = ["good", "great", "excellent", "happy", "love", "fantastic", "awesome"]
        negative_keywords = ["bad", "terrible", "poor", "hate", "unhappy", "awful", "disappointing"]

        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.5 + (positive_count / (len(text_lower.split()) + 1)) * 0.5 # Simulate higher score
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.5 - (negative_count / (len(text_lower.split()) + 1)) * 0.5 # Simulate lower score
        else:
            sentiment = "neutral"
            score = 0.5 # Explicitly set for clarity if counts are equal

        # Ensure score is within a plausible range
        score = round(max(0.0, min(1.0, score)), 2)

        result = {
            "text": data,
            "sentiment": sentiment,
            "score": score
        }

        logger.info(
            f"[{self.node_name}] Processed text: '{data[:50]}...' -> Sentiment: '{sentiment}', Score: {score}"
        )
        return result

# Example of how to use this node (for testing/demonstration outside Vishustra orchestration)
if __name__ == "__main__":
    import sys
    import os

    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock BaseNode's module path for standalone execution
    # In a real Vishustra setup, this import would resolve correctly.
    # This block is purely for local file execution without the full framework.
    if "vishustra_core.nodes.base_node" not in sys.modules:
        class MockBaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass
        BaseNode = MockBaseNode
        # Re-import SentimentAnalyzerNode to pick up the mock BaseNode if needed
        # For this specific case, if BaseNode is defined before SentimentAnalyzerNode,
        # it wouldn't need a re-import. Just ensuring the mock is in place.

    print("--- Testing SentimentAnalyzerNode ---")
    node = SentimentAnalyzerNode()
    test_context = {"user_id": "test_user", "timestamp": "2023-10-27T10:00:00Z"}

    # Test cases
    test_texts = [
        "This is an excellent product, I love it!",
        "The service was terrible and very disappointing.",
        "The weather is overcast with a chance of rain.",
        "I am so happy with the results.",
        "What a bad day.",
        "", # Empty string
        None # Non-string input
    ]

    for i, text_input in enumerate(test_texts):
        print(f"\nTest {i+1}: Input: '{text_input}'")
        try:
            result = node.process(text_input, test_context)
            print(f"Result: {result}")
        except (TypeError, ValueError) as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print("\n--- SentimentAnalyzerNode Testing Complete ---")