import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra node that performs sentiment analysis on input text.

    This node takes a string as input and returns a dictionary containing
    the detected sentiment label (e.g., 'positive', 'negative', 'neutral')
    and a simulated sentiment score.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The `data` input is expected to be a string. The method simulates
        sentiment analysis and returns a dictionary with 'sentiment' and 'score'.

        Args:
            data (Any): The input data, expected to be a string (text).
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used in this
                                       simulation but available for future extensions.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'sentiment' (str): The detected sentiment label.
                            - 'score' (float): A simulated sentiment score between -1.0 and 1.0.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"SentimentAnalyzer expects string input for analysis, "
                f"but received type: {type(data).__name__}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = data.lower().strip()

        if not text:
            logger.info("Received empty string for sentiment analysis. Returning neutral sentiment.")
            return {"sentiment": "neutral", "score": 0.0}

        # --- Simple keyword-based sentiment simulation ---
        # In a production system, this would integrate with an NLP library or model.
        positive_keywords = {
            "good", "great", "excellent", "happy", "love", "amazing",
            "wonderful", "fantastic", "pleasure", "joyful", "superb"
        }
        negative_keywords = {
            "bad", "terrible", "awful", "sad", "horrible", "hate",
            "unpleasant", "poor", "frustrating", "disappointing", "miserable"
        }

        # Split text into words and use a set for efficient lookup of unique words
        words = set(text.split())

        positive_count = sum(1 for word in words if word in positive_keywords)
        negative_count = sum(1 for word in words if word in negative_keywords)

        total_keywords_found = positive_count + negative_count

        sentiment_score: float
        sentiment_label: str

        if total_keywords_found == 0:
            # If no sentiment-bearing keywords are found, default to neutral
            sentiment_label = "neutral"
            sentiment_score = 0.0
        else:
            # Calculate a normalized score based on the counts
            sentiment_score = (positive_count - negative_count) / total_keywords_found

            # Determine the sentiment label based on score thresholds
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

        logger.debug(
            f"Analyzed text fragment '{text[:75]}{'...' if len(text) > 75 else ''}': "
            f"sentiment='{sentiment_label}', score={sentiment_score:.4f}"
        )

        return {"sentiment": sentiment_label, "score": round(sentiment_score, 4)}