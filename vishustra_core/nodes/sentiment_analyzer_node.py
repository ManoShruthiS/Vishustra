from typing import Any, Dict
import logging

# Assuming vishustra_core.nodes.base_node provides the BaseNode class
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node designed to perform sentiment analysis on input text.
    This node simulates sentiment detection using a simple keyword-based heuristic.
    In a production system, this would typically integrate with a dedicated
    natural language processing (NLP) library or an external sentiment analysis service.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Analyzes the sentiment of the input text data.

        Args:
            data (Any): The input data, expected to be a string of text for analysis.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node
                                       does not actively use context but adheres to
                                       the base node signature.

        Returns:
            Any: A dictionary containing the original text, the detected sentiment
                 (e.g., "positive", "negative", "neutral"), and a confidence score.

        Raises:
            ValueError: If the input `data` is not a string, as this node
                        is designed to process text.
            Exception: Catches and logs any unexpected errors during the sentiment
                       analysis process, then re-raises them to propagate failures.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise ValueError(
                f"Input for {self.node_name} must be a string, "
                f"but got {type(data).__name__}."
            )

        text_lower = data.lower()
        positive_score = 0
        negative_score = 0

        # Simple keyword lists for simulated sentiment analysis
        positive_keywords = ["good", "great", "excellent", "happy", "love", "joy",
                             "positive", "wonderful", "amazing", "fantastic", "brilliant"]
        negative_keywords = ["bad", "poor", "terrible", "sad", "hate", "anger",
                             "negative", "awful", "disappointing", "frustrating", "horrible"]

        try:
            for keyword in positive_keywords:
                positive_score += text_lower.count(keyword)

            for keyword in negative_keywords:
                negative_score += text_lower.count(keyword)

            sentiment = "neutral"
            if positive_score > negative_score:
                sentiment = "positive"
            elif negative_score > positive_score:
                sentiment = "negative"

            # Calculate a simple confidence score
            total_keyword_mentions = positive_score + negative_score
            confidence = 0.5 # Default for neutral or very weak signals
            if total_keyword_mentions > 0:
                # Confidence is higher for stronger dominance of one sentiment
                # This is a basic heuristic; real systems use more advanced models.
                confidence = abs(positive_score - negative_score) / total_keyword_mentions
                # Ensure confidence is within a reasonable range and not too low if keywords are found
                confidence = round(min(1.0, max(0.1, confidence + 0.1)), 2)
            else:
                # If no keywords found, confidence in neutral is moderate
                confidence = 0.6

            logger.info(
                f"[{self.node_name}] Processed text (first 50 chars): '{data[:50]}...' -> "
                f"Sentiment: '{sentiment}', Confidence: {confidence:.2f}"
            )

            return {
                "original_text": data,
                "sentiment": sentiment,
                "confidence": confidence
            }

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during sentiment analysis."
            )
            # Re-raise the exception after logging for upstream handling
            raise
