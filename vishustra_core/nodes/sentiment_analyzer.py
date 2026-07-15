import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that analyzes the sentiment of input text.
    It identifies sentiment (positive, negative, neutral) based on keyword matching
    and provides a basic sentiment score. This is a simulated analyzer for demonstration.
    """

    _POSITIVE_KEYWORDS = {
        "good", "great", "excellent", "happy", "love", "wonderful", "fantastic",
        "positive", "amazing", "best", "joy", "pleasure", "superb", "brilliant",
        "delightful", "optimistic", "favorable", "strong", "effective", "well"
    }
    _NEGATIVE_KEYWORDS = {
        "bad", "terrible", "awful", "hate", "poor", "miserable", "negative",
        "worst", "horrible", "disgusting", "sad", "anger", "frustration",
        "disappointing", "critical", "pessimistic", "unfavorable", "weak",
        "ineffective", "poorly"
    }

    def __init__(self) -> None:
        """
        Initializes the SentimentAnalyzerNode.
        """
        super().__init__()
        logger.debug(f"[{self.node_name}] Initialized sentiment analyzer node.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "sentiment_analyzer_node"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data (expected to be a string).

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.
                                       This node does not currently use the context for logic,
                                       but it is available for future enhancements.

        Returns:
            Dict[str, Any]: A dictionary containing the detected sentiment and score,
                            e.g., {'sentiment': 'positive', 'score': 0.75}.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, got {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.strip()
        if not text:
            error_msg = f"[{self.node_name}] Input text cannot be empty for sentiment analysis."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"[{self.node_name}] Analyzing sentiment for input text (first 75 chars): "
            f"'{text[:75]}{'...' if len(text) > 75 else ''}'"
        )

        # Basic tokenization: split by non-alphanumeric characters and convert to lowercase
        words = re.findall(r'\b\w+\b', text.lower())

        positive_count = 0
        negative_count = 0
        total_sentiment_bearing_words = 0

        for word in words:
            if word in self._POSITIVE_KEYWORDS:
                positive_count += 1
                total_sentiment_bearing_words += 1
            elif word in self._NEGATIVE_KEYWORDS:
                negative_count += 1
                total_sentiment_bearing_words += 1

        sentiment: str
        score: float = 0.0

        if total_sentiment_bearing_words > 0:
            score = (positive_count - negative_count) / total_sentiment_bearing_words
        else:
            # If no sentiment-bearing keywords are found, default to neutral.
            logger.debug(
                f"[{self.node_name}] No explicit sentiment-bearing keywords found. "
                "Defaulting to neutral sentiment."
            )

        # Determine sentiment based on score thresholds
        if score > 0.15:  # Tunable threshold for positive
            sentiment = "positive"
        elif score < -0.15:  # Tunable threshold for negative
            sentiment = "negative"
        else:
            sentiment = "neutral"

        result = {'sentiment': sentiment, 'score': round(score, 4)}
        logger.info(f"[{self.node_name}] Analysis complete. Sentiment: '{sentiment}', Score: {round(score, 4)}")
        return result