import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node dedicated to performing sentiment analysis on input text.

    This node processes a given string input, categorizing its sentiment
    as 'positive', 'negative', or 'neutral' based on a predefined set of keywords.
    It's designed to be a plug-and-play component within a larger orchestration
    workflow, enabling text-based insights.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this sentiment analysis node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Analyzes the sentiment of the input data, which is expected to be a string.

        The analysis is simulated using a simple keyword-matching mechanism.
        The result includes the original text and its determined sentiment.

        Args:
            data (Any): The input data for sentiment analysis, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. This node does not
                                       directly use context but adheres to the signature.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'original_text' (str): The input text that was analyzed.
                            - 'sentiment' (str): The classified sentiment ('positive',
                                                 'negative', or 'neutral').
                            - 'analysis_method' (str): Indicates the method used,
                                                       e.g., 'keyword_simulation'.

        Raises:
            ValueError: If the input 'data' is not a string, as this node
                        is specifically designed for text processing.
        """
        logger.info(f"[{self.node_name}] Initiating sentiment analysis for input data.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', but received '{type(data).__name__}'. "
                "Aborting sentiment analysis."
            )
            raise ValueError(
                f"SentimentAnalyzerNode expects string input, but got '{type(data).__name__}'."
            )

        text_lower = data.lower()
        sentiment = "neutral"

        # Define keywords for sentiment classification (simple simulation)
        positive_keywords = ["good", "great", "excellent", "love", "happy", "amazing", "fantastic", "wonderful", "like"]
        negative_keywords = ["bad", "terrible", "horrible", "hate", "unhappy", "awful", "poor", "dislike", "frustrating"]

        # Check for positive sentiment
        if any(keyword in text_lower for keyword in positive_keywords):
            sentiment = "positive"
        # Check for negative sentiment. This prioritizes positive if both types of keywords are present.
        # A more advanced real-world system would handle conflicting signals.
        elif any(keyword in text_lower for keyword in negative_keywords):
            sentiment = "negative"

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "analysis_method": "keyword_simulation"
        }

        logger.info(
            f"[{self.node_name}] Analysis complete. Input text classified as '{sentiment}'. "
            f"Original text snippet: '{data[:50]}...'"
        )
        return result