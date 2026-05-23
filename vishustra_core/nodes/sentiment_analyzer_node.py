import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.

    This node expects a string as input data and returns a dictionary
    containing the original text, a sentiment label (Positive, Negative, Neutral),
    and a sentiment score.
    """

    _positive_keywords: List[str] = [
        "good", "great", "excellent", "happy", "love", "awesome",
        "fantastic", "positive", "superb", "wonderful", "amazing"
    ]
    _negative_keywords: List[str] = [
        "bad", "terrible", "horrible", "sad", "hate", "awful",
        "dreadful", "negative", "poor", "unpleasant", "disappointing"
    ]

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        Args:
            data (Any): The input data, expected to be a string of text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing flow.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'original_text': The input text.
                            - 'sentiment': 'Positive', 'Negative', or 'Neutral'.
                            - 'score': A numerical sentiment score (positive for positive,
                                       negative for negative, near zero for neutral).

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the input string is empty.
        """
        if not isinstance(data, str):
            logger.error(f"SentimentAnalyzerNode expects string input, but received {type(data)}.")
            raise TypeError(f"SentimentAnalyzerNode: Input data must be a string, got {type(data)}")

        if not data.strip():
            logger.warning("SentimentAnalyzerNode received an empty or whitespace-only string, returning neutral sentiment.")
            return {
                "original_text": data,
                "sentiment": "Neutral",
                "score": 0.0
            }

        text_lower = data.lower()
        positive_score = sum(text_lower.count(word) for word in self._positive_keywords)
        negative_score = sum(text_lower.count(word) for word in self._negative_keywords)

        total_score = positive_score - negative_score
        
        sentiment_label: str
        if total_score > 0:
            sentiment_label = "Positive"
        elif total_score < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        logger.debug(f"Analyzed sentiment for text (first 50 chars): '{data[:50]}' -> {sentiment_label} (Score: {total_score})")

        return {
            "original_text": data,
            "sentiment": sentiment_label,
            "score": float(total_score)
        }