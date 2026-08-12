import logging
from typing import Any, Dict

# Assuming vishustra_core is a package and nodes.base_node is a module within it
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that performs sentiment analysis on input text.

    This node simulates sentiment detection (positive, negative, neutral)
    based on a simplified keyword matching approach. In a real-world scenario,
    it would integrate with an actual NLP model or service.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "Sentiment Analyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Analyzes the sentiment of the input text data.

        Expected input `data`: A string containing the text to analyze.
        Expected output: A dictionary containing the original text, detected sentiment
                         ('positive', 'negative', 'neutral'), and a simulated score.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used by this
                                       node's core logic but provided for extensibility.

        Returns:
            Any: A dictionary structured as follows:
                 `{"text": original_text, "sentiment": detected_sentiment, "score": simulated_score}`

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name} received invalid data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_to_analyze = data.strip()

        if not text_to_analyze:
            logger.warning(f"{self.node_name} received an empty string for analysis. Returning 'neutral' sentiment.")
            return {"text": data, "sentiment": "neutral", "score": 0.50}

        # --- Simulate sentiment analysis based on keywords ---
        lower_text = text_to_analyze.lower()
        sentiment = "neutral"
        score = 0.50  # Default neutral score

        positive_keywords = ["good", "great", "excellent", "wonderful", "amazing", "happy", "love", "awesome", "perfect"]
        negative_keywords = ["bad", "terrible", "poor", "awful", "horrible", "sad", "hate", "disappointing", "frustrating"]

        positive_count = sum(1 for keyword in positive_keywords if keyword in lower_text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in lower_text)

        if positive_count > negative_count:
            sentiment = "positive"
            # Simulate a higher score for positive sentiment, capped near 0.99
            score = 0.50 + min(0.49, positive_count * 0.15)
        elif negative_count > positive_count:
            sentiment = "negative"
            # Simulate a lower score for negative sentiment, floored near 0.01
            score = 0.50 - min(0.49, negative_count * 0.15)
        # If counts are equal or zero, sentiment remains 'neutral' and score 0.50

        # Log the result for debugging purposes
        log_text_preview = text_to_analyze[:70] + ("..." if len(text_to_analyze) > 70 else "")
        logger.debug(
            f"Analyzed text '{log_text_preview}': "
            f"Detected sentiment: '{sentiment}' (Score: {score:.2f})"
        )

        return {"text": data, "sentiment": sentiment, "score": round(score, 2)}
