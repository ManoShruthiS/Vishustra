import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that analyzes the sentiment of a given text string.
    It simulates sentiment analysis, categorizing text as 'positive', 'negative', or 'neutral'.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to determine its sentiment.

        The node expects 'data' to be a string. It will attempt to classify
        the sentiment as 'positive', 'negative', or 'neutral' based on simple
        keyword matching.

        Args:
            data: The input text string to be analyzed.
            context: A dictionary containing contextual information for processing.
                     (Currently not used for sentiment determination, but available
                     for future enhancements like model configuration.)

        Returns:
            A string indicating the determined sentiment: "positive", "negative", or "neutral".

        Raises:
            ValueError: If the input 'data' is not a string.
            Exception: For unexpected errors during processing.
        """
        logger.debug(f"[{self.node_name}] Starting sentiment analysis for data: {data}")
        logger.debug(f"[{self.node_name}] Context received: {context}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input type. Expected string, got {type(data).__name__}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = data.lower()
        sentiment = "neutral"

        try:
            positive_keywords = ["good", "great", "excellent", "happy", "love", "positive", "fantastic", "awesome"]
            negative_keywords = ["bad", "terrible", "horrible", "sad", "hate", "negative", "awful", "poor"]

            if any(keyword in text for keyword in positive_keywords):
                sentiment = "positive"
            elif any(keyword in text for keyword in negative_keywords):
                sentiment = "negative"

            logger.info(f"[{self.node_name}] Analyzed sentiment for '{data[:50]}...': {sentiment}")
            return sentiment
        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during sentiment analysis: {e}"
            logger.exception(error_msg)
            raise Exception(error_msg) from e