import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

_logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that simulates sentiment analysis on input text.

    This node expects a string as input data and returns a dictionary
    containing the original text, the detected sentiment (positive, negative, neutral),
    and a simulated confidence score.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data: The input data, expected to be a string containing text
                  to be analyzed for sentiment.
            context: A dictionary containing contextual information
                     that might be relevant for processing (not used by this node).

        Returns:
            A dictionary with the sentiment analysis result:
            {
                "text": str,          # The original input text
                "sentiment": str,     # 'positive', 'negative', or 'neutral'
                "score": float        # A simulated confidence score for the sentiment (0.0 to 1.0)
            }

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            _logger.error(
                "[%s] Received non-string data of type %s. Expected string for sentiment analysis.",
                self.node_name, type(data)
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a string for sentiment analysis, "
                f"but received type {type(data)}."
            )

        text_lower = data.lower()
        sentiment_result: Dict[str, Any] = {"text": data}

        # Simulate sentiment analysis based on keywords
        if any(keyword in text_lower for keyword in ["good", "great", "excellent", "happy", "positive"]):
            sentiment_result["sentiment"] = "positive"
            sentiment_result["score"] = 0.85  # Simulated score
        elif any(keyword in text_lower for keyword in ["bad", "terrible", "poor", "sad", "negative"]):
            sentiment_result["sentiment"] = "negative"
            sentiment_result["score"] = 0.92  # Simulated score
        else:
            sentiment_result["sentiment"] = "neutral"
            sentiment_result["score"] = 0.55  # Simulated score

        _logger.debug(
            "[%s] Analyzed text (first 50 chars): '%s' -> Sentiment: %s (score: %.2f)",
            self.node_name, data[:50], sentiment_result["sentiment"], sentiment_result["score"]
        )

        return sentiment_result