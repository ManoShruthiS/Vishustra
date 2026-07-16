import logging
from typing import Any, Dict

# Simulating the import path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.

    This node simulates sentiment analysis by identifying positive and negative
    keywords within the input string. It returns a dictionary containing the
    original text and its determined sentiment.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information for the processing pipeline.
                                      (Not directly used by this node, but part
                                      of the required signature).

        Returns:
            Dict[str, Any]: A dictionary containing the original text and
                            its determined sentiment (e.g., "positive",
                            "negative", "neutral").

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for sentiment analysis."
            )

        text_to_analyze = data.lower()
        sentiment = "neutral"

        if not text_to_analyze.strip():
            logger.info(f"[{self.node_name}] Received empty or whitespace-only text. Assigning neutral sentiment.")
            return {"text": data, "sentiment": "neutral", "reason": "empty_input"}

        # Simple keyword-based sentiment simulation
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic", "positive"]
        negative_keywords = ["bad", "terrible", "horrible", "sad", "hate", "awful", "unpleasant", "negative"]

        positive_score = sum(1 for keyword in positive_keywords if keyword in text_to_analyze)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text_to_analyze)

        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        else:
            sentiment = "neutral" # Default or if scores are equal

        logger.debug(
            f"[{self.node_name}] Analyzed text snippet: '{data[:50]}{'...' if len(data) > 50 else ''}' -> Sentiment: {sentiment}"
        )

        return {"text": data, "sentiment": sentiment}

if __name__ == '__main__':
    # This block demonstrates basic usage and is not part of the core node
    # It's here for local testing purposes.
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG) # Enable debug logging for this example

    analyzer = SentimentAnalyzer()

    test_cases = [
        "This is an absolutely fantastic product!",
        "I had a terrible experience with their customer service.",
        "The weather today is just okay, nothing special.",
        "I love this new feature, it's great!",
        "Why is everything so bad and awful?",
        "Neutral statement with no strong feelings.",
        "",
        "   ",
        123, # This should raise a TypeError
        {"key": "value"} # This should raise a TypeError
    ]

    for i, test_data in enumerate(test_cases):
        logger.info(f"\n--- Test Case {i+1} ---")
        try:
            result = analyzer.process(test_data, {})
            logger.info(f"Input: '{test_data}'\nResult: {result}")
        except Exception as e:
            logger.error(f"Error processing '{test_data}': {e}")
