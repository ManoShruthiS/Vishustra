import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs a basic sentiment analysis
    on input text data.

    This node identifies a general sentiment (Positive, Negative, Neutral)
    based on a simplified keyword matching algorithm. It's designed for
    initial text evaluation within an orchestration pipeline.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The method expects the `data` parameter to be a string. If not, it logs a warning
        and returns a structured error message within the result dictionary.
        Sentiment is determined by summing scores from predefined positive and negative keywords.

        Args:
            data: The input data, typically expected to be a string containing text
                  to be analyzed.
            context: A dictionary containing contextual information for the processing.
                     This node does not currently utilize the context, but it is
                     available for future extensions.

        Returns:
            A dictionary containing:
            - "original_text": The input data.
            - "sentiment": The detected sentiment label ("Positive", "Negative", "Neutral", or "Error").
            - "score": A numerical representation of the sentiment (e.g., sum of keyword scores).
            - "error": (Optional) An error message if input validation fails.

            Example of a successful return:
            {
                "original_text": "This product is absolutely fantastic!",
                "sentiment": "Positive",
                "score": 2.5
            }
            Example of an error return:
            {
                "original_text": 123,
                "sentiment": "Error",
                "score": 0.0,
                "error": "Input data must be a string."
            }
        """
        logger.debug(f"[{self.node_name}] Starting processing. Input type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input type. Expected string, got {type(data).__name__}. " \
                        "Returning an error state for sentiment analysis."
            logger.warning(error_msg)
            return {
                "original_text": data,
                "sentiment": "Error",
                "score": 0.0,
                "error": "Input data must be a string."
            }

        text_lower = data.lower()
        sentiment_score = 0.0
        found_sentiment_keywords = False

        # Simple keyword lists for sentiment analysis. Scores are additive.
        positive_keywords = {
            "good": 1.0, "great": 1.5, "excellent": 2.0, "happy": 1.0, "love": 1.5,
            "fantastic": 2.0, "amazing": 1.8, "wonderful": 1.5, "brilliant": 1.5,
            "positive": 1.0
        }
        negative_keywords = {
            "bad": -1.0, "terrible": -1.5, "poor": -1.0, "unhappy": -1.0, "hate": -1.5,
            "awful": -2.0, "horrible": -1.8, "disappointing": -1.5, "frustrating": -1.5,
            "negative": -1.0
        }

        # Accumulate scores from positive keywords
        for word, score in positive_keywords.items():
            if word in text_lower:
                sentiment_score += score
                found_sentiment_keywords = True

        # Accumulate scores from negative keywords
        for word, score in negative_keywords.items():
            if word in text_lower:
                sentiment_score += score
                found_sentiment_keywords = True

        sentiment_label: str
        # Determine the final sentiment label based on the accumulated score
        if not found_sentiment_keywords:
            sentiment_label = "Neutral"
            # Ensure score is exactly 0.0 if no specific sentiment keywords were found
            sentiment_score = 0.0
        elif sentiment_score > 0.5:  # Threshold for positive sentiment
            sentiment_label = "Positive"
        elif sentiment_score < -0.5: # Threshold for negative sentiment
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral" # For scores between -0.5 and 0.5, or mixed sentiments cancelling out

        result = {
            "original_text": data,
            "sentiment": sentiment_label,
            "score": sentiment_score
        }

        logger.debug(f"[{self.node_name}] Finished processing. Identified sentiment: '{sentiment_label}' "
                     f"(Score: {sentiment_score:.2f}) for text length {len(data)}.")
        return result