import logging
from typing import Any, Dict

# Assuming vishustra_core is a package at the root of the project
# For local testing, you might need to adjust this or mock it.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on text data.

    This node simulates sentiment analysis, identifying the emotional tone
    (positive, negative, or neutral) of the input text.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data: The input data, expected to be a string containing text.
            context: A dictionary containing contextual information for the process.

        Returns:
            A dictionary containing the original text, its determined sentiment,
            and a simulated sentiment score.

            Example:
            {
                "text": "This product is absolutely fantastic!",
                "sentiment": "positive",
                "score": 0.95
            }

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type for sentiment analysis. "
                f"Expected str, got {type(data).__name__}."
            )
            raise TypeError(
                f"SentimentAnalyzerNode expects 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty string for analysis.")
            return {
                "text": data,
                "sentiment": "neutral",
                "score": 0.0
            }

        text_lower = data.lower()
        
        # Simulated sentiment analysis logic
        positive_keywords = ["good", "great", "excellent", "fantastic", "love", "happy", "amazing", "wonderful"]
        negative_keywords = ["bad", "terrible", "horrible", "awful", "hate", "unhappy", "poor", "frustrating"]
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        sentiment = "neutral"
        score = 0.0

        if pos_count > neg_count:
            sentiment = "positive"
            # Simulate a score based on strength
            score = min(0.5 + pos_count * 0.1, 0.99)
            logger.info(f"[{self.node_name}] Detected positive sentiment.")
        elif neg_count > pos_count:
            sentiment = "negative"
            # Simulate a score based on strength
            score = max(-0.5 - neg_count * 0.1, -0.99)
            logger.info(f"[{self.node_name}] Detected negative sentiment.")
        else:
            sentiment = "neutral"
            score = 0.0
            logger.info(f"[{self.node_name}] Detected neutral sentiment or mixed signals.")
        
        # The score can be interpreted as a continuum from -1.0 (strong negative)
        # to 1.0 (strong positive), with 0.0 being neutral.
        
        result = {
            "text": data,
            "sentiment": sentiment,
            "score": score
        }
        
        logger.debug(f"[{self.node_name}] Processed text '{data[:50]}...' -> {result}")
        return result

# Example of how to use this node (for demonstration, not part of the required output)
if __name__ == '__main__':
    # Basic logging configuration for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    analyzer = SentimentAnalyzerNode()
    
    # Test cases
    test_cases = [
        "This product is absolutely fantastic! I love it.",
        "The service was terrible and I am very unhappy.",
        "The weather is just okay today.",
        "It was a good experience, but not great.",
        "",
        123, # Incorrect type
        "What a wonderful day for a great adventure!"
    ]
    
    for i, test_data in enumerate(test_cases):
        try:
            print(f"\n--- Test Case {i+1} ---")
            print(f"Input: {test_data!r}")
            output = analyzer.process(test_data, {})
            print(f"Output: {output}")
        except Exception as e:
            print(f"Error processing '{test_data!r}': {e}")
            logger.exception(f"Exception during processing of test case {i+1}")