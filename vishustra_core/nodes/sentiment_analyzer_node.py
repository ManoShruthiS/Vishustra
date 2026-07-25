import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that performs simulated sentiment analysis on input text.

    This node processes a given string, attempting to classify its sentiment
    as positive, negative, mixed, or neutral based on a simple keyword matching
    mechanism. It's designed to illustrate data transformation within the framework.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string) to determine its sentiment.

        The sentiment analysis is simulated using a basic keyword-matching algorithm.
        This method ensures robust handling of input types and provides a structured
        output including the detected sentiment and a simulated confidence score.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        to be analyzed for sentiment.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     relevant to the current processing flow.
                                     This node does not directly use context but
                                     receives it as per BaseNode contract.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis result, with keys
                            "sentiment" (e.g., "positive", "negative", "neutral", "mixed"),
                            "confidence" (float between 0.0 and 1.0), and "analyzed_text".

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"SentimentAnalyzerNode received invalid data type: {type(data)}. "
                "Expected a string for sentiment analysis."
            )
            raise TypeError(
                f"SentimentAnalyzerNode requires string data for processing, "
                f"but received {type(data)}."
            )

        text = data.strip()
        if not text:
            logger.warning("SentimentAnalyzerNode received an empty or whitespace-only string. "
                           "Returning neutral sentiment.")
            return {"sentiment": "neutral", "confidence": 0.5, "analyzed_text": data}

        sentiment = "neutral"
        confidence = 0.5 # Default confidence for neutral

        # Simple keyword-based sentiment simulation
        positive_keywords = {"good", "great", "excellent", "happy", "love", "awesome", "fantastic", "positive", "joy"}
        negative_keywords = {"bad", "terrible", "awful", "unhappy", "hate", "poor", "negative", "horrible", "sad"}

        normalized_text = text.lower()
        words = set(normalized_text.split()) # Use set for faster lookups

        is_positive = any(word in words for word in positive_keywords)
        is_negative = any(word in words for word in negative_keywords)

        if is_positive and not is_negative:
            sentiment = "positive"
            confidence = 0.9
        elif is_negative and not is_positive:
            sentiment = "negative"
            confidence = 0.9
        elif is_positive and is_negative:
            # If both positive and negative keywords are present
            sentiment = "mixed"
            confidence = 0.7
        else:
            sentiment = "neutral"
            confidence = 0.6 # Slightly higher than default if processed successfully without strong signals

        result = {"sentiment": sentiment, "confidence": confidence, "analyzed_text": data}
        logger.debug(f"Sentiment analysis for text '{data[:50]}...': {result}")
        return result
