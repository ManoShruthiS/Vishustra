import logging
from typing import Any, Dict

# Assuming the project structure places base_node in vishustra_core.nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that analyzes the sentiment of input text.

    This node simulates sentiment analysis by classifying text into
    'positive', 'negative', or 'neutral' categories based on keyword matching.
    It expects a string as input data and enriches it with a sentiment label.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string) and determines its sentiment.

        Args:
            data: The input text to be analyzed for sentiment.
                  Expected type: str.
            context: A dictionary containing contextual information, such as configuration
                     or runtime variables. Not directly used for sentiment determination
                     in this basic implementation but available for future extensions.

        Returns:
            A dictionary containing the original 'text' and its derived 'sentiment'
            ('positive', 'negative', 'neutral').

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Attempting to analyze sentiment for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input type. Expected 'str' for sentiment analysis, "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        stripped_data = data.strip()
        if not stripped_data:
            error_msg = f"[{self.node_name}] Input text cannot be empty or consist only of whitespace characters."
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_lower = stripped_data.lower()
        sentiment = "neutral"

        # Simple keyword-based sentiment analysis simulation
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic", "amazing", "superb", "brilliant", "positive"]
        negative_keywords = ["bad", "terrible", "horrible", "sad", "hate", "awful", "disappointing", "poor", "negative", "dreadful"]

        found_positive = any(keyword in text_lower for keyword in positive_keywords)
        found_negative = any(keyword in text_lower for keyword in negative_keywords)

        if found_positive and not found_negative:
            sentiment = "positive"
        elif found_negative and not found_positive:
            sentiment = "negative"
        elif found_positive and found_negative:
            # If both types of keywords are present, consider it mixed or neutral for this basic simulation
            sentiment = "neutral"
        # Otherwise, if neither type of keyword is found, it remains "neutral"

        logger.debug(f"[{self.node_name}] Successfully analyzed text sentiment: '{sentiment}' for text snippet: '{stripped_data[:50]}...'")

        return {
            "text": stripped_data,
            "sentiment": sentiment
        }
