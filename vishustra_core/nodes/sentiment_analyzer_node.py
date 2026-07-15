import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that performs sentiment analysis on input text data.

    This node simulates sentiment analysis by checking for the presence of
    predefined positive and negative keywords within the input text.
    It returns a dictionary indicating the sentiment ('positive', 'negative', 'neutral')
    and a corresponding score.
    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzerNode with predefined keyword lists
        for sentiment detection. In a real-world scenario, this might involve
        loading an NLP model or an external API client.
        """
        self._positive_keywords = {
            "great", "excellent", "love", "happy", "good", "amazing",
            "fantastic", "wonderful", "brilliant", "joyful", "positive"
        }
        self._negative_keywords = {
            "bad", "terrible", "hate", "sad", "poor", "awful",
            "horrible", "frustrating", "disappointing", "angry", "negative"
        }
        logger.info(f"[{self.node_name}] Initialized with simulated sentiment keywords.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "SentimentAnalyzerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.

        The node expects 'data' to be a string. It counts occurrences of
        positive and negative keywords to determine the overall sentiment.

        Args:
            data: The input text data (expected to be a string) to analyze.
            context: A dictionary of context variables, not directly used in
                     this specific node's logic but available for more complex scenarios.

        Returns:
            A dictionary with two keys:
            - 'sentiment': A string ('positive', 'negative', 'neutral') indicating the detected sentiment.
            - 'score': A float representing the normalized sentiment strength (positive for positive,
                       negative for negative, 0 for neutral or mixed). The score is between -1.0 and 1.0.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to perform sentiment analysis."
            )
            logger.error(error_msg)
            # For robustness, we can choose to raise an error or return a neutral/error state.
            # Returning a structured error state for node compatibility.
            return {"sentiment": "error", "score": 0.0, "error_message": error_msg}

        text_lower = data.lower()
        positive_count = sum(1 for keyword in self._positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self._negative_keywords if keyword in text_lower)

        total_sentiment_words = positive_count + negative_count

        sentiment: str
        score: float = 0.0

        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
            logger.debug(f"[{self.node_name}] Processed: '{data[:75]}{'...' if len(data) > 75 else ''}' -> {sentiment} (score: {score:.2f})")
        elif positive_count > negative_count:
            sentiment = "positive"
            # Normalize score between 0.0 and 1.0 based on difference
            score = (positive_count - negative_count) / total_sentiment_words
            logger.debug(f"[{self.node_name}] Processed: '{data[:75]}{'...' if len(data) > 75 else ''}' -> {sentiment} (score: {score:.2f})")
        elif negative_count > positive_count:
            sentiment = "negative"
            # Normalize score between -1.0 and 0.0 based on difference
            score = (positive_count - negative_count) / total_sentiment_words
            logger.debug(f"[{self.node_name}] Processed: '{data[:75]}{'...' if len(data) > 75 else ''}' -> {sentiment} (score: {score:.2f})")
        else:  # Equal counts or mixed bag leading to neutral
            sentiment = "neutral"
            score = 0.0
            logger.debug(f"[{self.node_name}] Processed: '{data[:75]}{'...' if len(data) > 75 else ''}' -> {sentiment} (score: {score:.2f}) (mixed/equal sentiment words)")

        return {"sentiment": sentiment, "score": score}