import logging
from typing import Any, Dict

# Assuming the project structure places BaseNode in vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that analyzes the sentiment of input text.
    It simulates sentiment analysis by identifying positive and negative keywords
    within the text and computing a basic sentiment score.
    """

    # These keyword lists are simplified for demonstration purposes.
    # In a real-world scenario, this would involve more sophisticated NLP models or APIs.
    _POSITIVE_KEYWORDS = {"good", "great", "excellent", "happy", "love", "awesome", "fantastic", "positive", "super", "amazing", "wonderful"}
    _NEGATIVE_KEYWORDS = {"bad", "terrible", "horrible", "sad", "hate", "awful", "negative", "poor", "frustrating", "disappointing"}
    
    _NEUTRAL_THRESHOLD = 0.2 # Score magnitude below which sentiment is considered neutral

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        Expected `data` type: str (the text content to be analyzed).
        The `context` dictionary is provided by the framework but is not directly
        utilized by this specific node's logic for sentiment determination.

        Returns:
            A dictionary containing the original text, the detected sentiment
            ('positive', 'negative', 'neutral'), and a calculated sentiment score.
            Example: {"text": "The service was excellent!", "sentiment": "positive", "score": 0.8}

        Raises:
            TypeError: If the input `data` is not a string, indicating an
                       unsupported data format for text analysis.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for input data.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str' for text analysis, "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_lower = data.lower()
        words = text_lower.split()

        positive_count = sum(1 for word in words if word in self._POSITIVE_KEYWORDS)
        negative_count = sum(1 for word in words if word in self._NEGATIVE_KEYWORDS)

        total_sentiment_bearing_words = positive_count + negative_count

        sentiment = "neutral"
        score = 0.0

        if total_sentiment_bearing_words > 0:
            # Calculate a raw score normalized by the number of sentiment-bearing words
            raw_score = (positive_count - negative_count) / total_sentiment_bearing_words
            score = round(raw_score, 4) # Round for consistent output

            if score >= self._NEUTRAL_THRESHOLD:
                sentiment = "positive"
            elif score <= -self._NEUTRAL_THRESHOLD:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            logger.info(f"[{self.node_name}] No explicit sentiment-bearing keywords found in the text.")

        result = {
            "text": data,  # Preserve and return the original text
            "sentiment": sentiment,
            "score": score
        }

        logger.debug(
            f"[{self.node_name}] Successfully analyzed text (excerpt: '{data[:75]}...') "
            f"-> Sentiment: {sentiment}, Score: {score}"
        )
        return result