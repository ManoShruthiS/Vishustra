import logging
from typing import Any, Dict

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on input text.
    It categorizes text into 'positive', 'negative', or 'neutral' based on simple
    keyword detection.
    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzerNode.
        In a real-world scenario, this would load an NLP model or client.
        """
        logger.debug("SentimentAnalyzerNode initialized, ready for text analysis.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data, expected to be a string containing text
                        for sentiment analysis.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current execution flow.

        Returns:
            Dict[str, Any]: A dictionary containing the original text and its
                            determined sentiment ('positive', 'negative', 'neutral').

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_to_analyze = data.lower()
        sentiment = "neutral"

        # Simulate sentiment detection using a simple keyword-based approach
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic"]
        negative_keywords = ["bad", "terrible", "awful", "sad", "hate", "poor", "disappointing"]

        if any(keyword in text_to_analyze for keyword in positive_keywords):
            sentiment = "positive"
        elif any(keyword in text_to_analyze for keyword in negative_keywords):
            sentiment = "negative"
        # If no strong positive or negative indicators, it defaults to neutral

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "node_processed_by": self.node_name
        }

        logger.info(
            f"[{self.node_name}] Processed text (first 50 chars): "
            f"'{data[:50]}{'...' if len(data) > 50 else ''}' -> Sentiment: '{sentiment}'"
        )
        return result