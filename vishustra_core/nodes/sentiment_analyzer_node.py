import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that performs sentiment analysis on input text.

    This node simulates sentiment analysis by identifying key positive,
    negative, or neutral terms within the input string to determine
    an overall sentiment.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        Expected `data` type: `str` (the text to analyze).
        The `context` dictionary can be used to pass configuration,
        e.g., specific sentiment models or language settings in a
        more advanced implementation. For this simulation, it's not
        directly utilized beyond logging.

        Returns a dictionary containing:
        - 'original_text': The input text.
        - 'sentiment': 'positive', 'negative', or 'neutral'.
        - 'confidence': A simulated confidence score (float between 0 and 1).

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an unexpected issue occurs during analysis (though
                        less likely in this simple simulation).
        """
        if not isinstance(data, str):
            error_msg = (
                f"SentimentAnalyzerNode '{self.node_name}' expects string input, "
                f"but received type: {type(data).__name__}. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.lower()  # Convert to lowercase for case-insensitive matching
        sentiment = "neutral"
        confidence = 0.5  # Base confidence for neutral sentiment

        # Simple keyword-based sentiment simulation
        positive_keywords = [
            "good", "great", "excellent", "wonderful", "amazing",
            "happy", "love", "awesome", "fantastic", "superb"
        ]
        negative_keywords = [
            "bad", "terrible", "poor", "horrible", "awful",
            "sad", "hate", "disappointing", "frustrating", "miserable"
        ]

        positive_score = sum(1 for keyword in positive_keywords if keyword in text)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text)

        if positive_score > negative_score:
            sentiment = "positive"
            # Simulate higher confidence for stronger positive signals
            confidence = min(0.95, 0.6 + (positive_score * 0.1))
        elif negative_score > positive_score:
            sentiment = "negative"
            # Simulate higher confidence for stronger negative signals
            confidence = min(0.95, 0.6 + (negative_score * 0.1))
        else:
            # If scores are equal or both zero, it leans neutral
            # Higher confidence if genuinely neutral (no strong signals)
            if positive_score == 0 and negative_score == 0:
                confidence = 0.75
            # Otherwise, it's neutral with some conflicting or weak signals
            else:
                confidence = 0.55

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)  # Round for cleaner output
        }

        logger.info(
            f"Node '{self.node_name}' successfully processed text and detected "
            f"sentiment: '{sentiment}' (Confidence: {confidence:.2f}). Context: {context}"
        )
        return result