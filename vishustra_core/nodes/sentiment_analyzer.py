import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra processing node designed to analyze the sentiment of input text.
    It categorizes text sentiment as positive, negative, or neutral and provides
    a corresponding confidence score.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the provided text data.

        This method expects `data` to be a string. It simulates sentiment analysis
        by checking for specific keywords. In a production environment, this would
        integrate with a robust NLP library or a dedicated sentiment analysis API.

        Args:
            data (Any): The text string for which sentiment is to be analyzed.
                        Expected type: str.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node
                                       does not currently use specific context values
                                       but adheres to the interface.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "original_text": The input text.
                            - "sentiment": The detected sentiment ('positive', 'negative', 'neutral').
                            - "score": A simulated confidence score for the detected sentiment
                                       (float between 0.0 and 1.0).

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty or whitespace-only string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"SentimentAnalyzer received invalid data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Use .strip() to consider strings with only whitespace as empty
        if not data.strip():
            error_msg = "SentimentAnalyzer received an empty or whitespace-only string for analysis."
            logger.error(error_msg)
            raise ValueError(error_msg)

        processed_text = data.lower()
        sentiment = "neutral"
        score = 0.5  # Default for neutral sentiment

        # Simulate sentiment detection based on simple keyword matching.
        # This is a placeholder for a more sophisticated NLP model or external service.
        positive_keywords = ["great", "excellent", "happy", "love", "awesome", "fantastic", "good"]
        negative_keywords = ["bad", "terrible", "sad", "hate", "awful", "horrible", "poor"]

        if any(keyword in processed_text for keyword in positive_keywords):
            sentiment = "positive"
            score = 0.85 # High confidence for positive
        elif any(keyword in processed_text for keyword in negative_keywords):
            sentiment = "negative"
            score = 0.15 # High confidence for negative
        # If both or neither, it remains neutral or the first match wins.
        # For a more nuanced approach, one would use compound scores or actual models.

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "score": score
        }

        # Log a summary of the processing for observability
        log_text_preview = data[:70] + "..." if len(data) > 70 else data
        logger.info(
            f"Sentiment analysis completed for text: '{log_text_preview}'. "
            f"Detected sentiment: '{sentiment}' with score: {score:.2f}."
        )

        return result
