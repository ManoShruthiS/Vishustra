import logging
from typing import Any, Dict

# Import BaseNode from the core framework as specified
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node designed to perform sentiment analysis on text input.

    This node simulates sentiment detection by analyzing the presence of
    predefined positive and negative keywords within the input text. It
    returns a sentiment label (positive, negative, neutral, or N/A) along
    with a numerical score indicating the strength of the sentiment.
    """

    # Predefined sets of keywords for simulated sentiment detection.
    # In a real-world scenario, this would involve integrating with an NLP library
    # or a dedicated sentiment analysis model/API.
    _POSITIVE_WORDS = {
        "good", "great", "excellent", "happy", "love", "awesome", "fantastic",
        "wonderful", "amazing", "superb", "brilliant", "joy", "success"
    }
    _NEGATIVE_WORDS = {
        "bad", "terrible", "poor", "sad", "hate", "awful", "horrible",
        "frustrating", "disappointing", "failure", "unhappy", "stress"
    }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The node expects the `data` parameter to be a string containing
        the text to be analyzed. It calculates a simple sentiment score
        based on the occurrence of positive and negative keywords.

        Args:
            data: The text content (string) to be analyzed for sentiment.
            context: A dictionary containing execution-specific context. While not
                     directly used by this node, it adheres to the BaseNode interface
                     and allows for future extensibility or dynamic configuration.

        Returns:
            A dictionary containing the sentiment analysis result:
            - 'sentiment': A string ("positive", "negative", "neutral", "N/A") indicating
                           the overall sentiment.
            - 'score': A float representing the calculated sentiment score
                       (positive_word_count - negative_word_count).
            - 'details': A dictionary providing granular breakdown, including
                         'positive_count', 'negative_count', and 'neutral_count'.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        logger.debug(f"SentimentAnalyzerNode received data of type: {type(data).__name__}")

        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for SentimentAnalyzerNode. "
                f"Expected 'str', but received '{type(data).__name__}'. Data: {data!r}"
            )
            raise TypeError(
                f"SentimentAnalyzerNode expects string input for analysis, "
                f"received {type(data).__name__}."
            )

        text_cleaned = data.strip().lower()

        if not text_cleaned:
            logger.warning("SentimentAnalyzerNode received an empty or whitespace-only string for analysis.")
            return {
                "sentiment": "N/A",
                "score": 0.0,
                "details": {"positive_count": 0, "negative_count": 0, "neutral_count": 0}
            }

        positive_count = 0
        negative_count = 0
        words = text_cleaned.split()
        total_words = len(words)

        for word in words:
            if word in self._POSITIVE_WORDS:
                positive_count += 1
            elif word in self._NEGATIVE_WORDS:
                negative_count += 1

        sentiment_score = float(positive_count - negative_count)

        if sentiment_score > 0:
            sentiment_label = "positive"
        elif sentiment_score < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        neutral_count = total_words - positive_count - negative_count
        # Ensure neutral count isn't negative due to overlapping words/non-exclusive sets
        neutral_count = max(0, neutral_count)

        logger.info(
            f"Analyzed text segment (first 50 chars: '{data[:50]}...'). "
            f"Determined sentiment: '{sentiment_label}' with score: {sentiment_score:.2f}."
        )
        logger.debug(
            f"Sentiment breakdown: Positive={positive_count}, Negative={negative_count}, Neutral={neutral_count}."
        )

        return {
            "sentiment": sentiment_label,
            "score": sentiment_score,
            "details": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count
            }
        }