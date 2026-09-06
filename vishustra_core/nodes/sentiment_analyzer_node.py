import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that analyzes the sentiment of input text data.

    This node simulates sentiment analysis by identifying keywords within the
    input text and categorizing the sentiment as positive, negative, or neutral.
    It expects a string as input and returns a dictionary containing the
    original text, the detected sentiment, and a placeholder score.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        Expected `data` type: `str`.
        Expected `context` keys: None specific for this node's core logic,
        but available for future extensions or metadata.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the current processing flow.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results, with keys:
                            - "text": The original input text.
                            - "sentiment": "positive", "negative", or "neutral".
                            - "score": A placeholder float representing sentiment strength.
                            - "error": (Optional) An error message if processing failed.
        """
        if not isinstance(data, str):
            error_msg = f"SentimentAnalyzerNode received non-string data. Expected string, got {type(data).__name__}."
            logger.warning(error_msg)
            return {
                "text": str(data),
                "sentiment": "error",
                "score": 0.0,
                "error": error_msg
            }

        text = data.lower()
        sentiment = "neutral"
        score = 0.0

        positive_keywords = ["great", "excellent", "good", "happy", "love", "awesome", "positive", "fine", "super"]
        negative_keywords = ["bad", "terrible", "poor", "sad", "hate", "awful", "negative", "horrible", "ugly"]

        has_positive = any(keyword in text for keyword in positive_keywords)
        has_negative = any(keyword in text for keyword in negative_keywords)

        if has_negative and not has_positive:
            sentiment = "negative"
            score = -0.8  # Placeholder score
        elif has_positive and not has_negative:
            sentiment = "positive"
            score = 0.8   # Placeholder score
        elif has_positive and has_negative:
            # For simplicity, if both positive and negative keywords are present,
            # we can make a heuristic decision or mark as mixed/neutral.
            # Here, we'll lean towards negative if present as a common simple model strategy.
            sentiment = "negative" if has_negative else "positive"
            score = -0.1 if has_negative else 0.1
            if not has_positive and not has_negative:
                sentiment = "neutral"
                score = 0.0
            logger.debug(f"Sentiment analysis for '{data[:50]}...' detected both positive and negative keywords, defaulting to '{sentiment}'.")
        else:
            sentiment = "neutral"
            score = 0.0

        logger.info(f"Analyzed sentiment for '{data[:50]}...' as '{sentiment}' with score {score:.2f}.")

        return {
            "text": data,
            "sentiment": sentiment,
            "score": score
        }