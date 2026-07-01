import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to analyze the sentiment of a given text input.

    This node simulates sentiment analysis using a basic keyword-matching approach.
    In a production environment, this would typically integrate with a dedicated
    Natural Language Processing (NLP) service or a sophisticated language model
    for more robust and nuanced sentiment detection.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its underlying sentiment.

        Args:
            data: The input data, which is expected to be a string representing text.
            context: A dictionary containing contextual information relevant for processing.
                     (Currently not utilized for the sentiment logic, but available
                     for future enhancements such as configuration parameters).

        Returns:
            A dictionary containing:
            - "text": The original input text.
            - "sentiment": The analyzed sentiment ("Positive", "Negative", or "Neutral").
            - "score": A numerical confidence score for the detected sentiment,
                       ranging approximately from -1.0 (highly negative) to 1.0 (highly positive).

        Raises:
            ValueError: If the input `data` is not a string, as this node is designed
                        specifically for text processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data!r}"
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        # Handle empty or whitespace-only strings gracefully
        if not data.strip():
            logger.info(
                f"[{self.node_name}] Received empty or whitespace-only string. "
                "Defaulting to 'Neutral' sentiment."
            )
            return {"text": data, "sentiment": "Neutral", "score": 0.0}

        logger.info(f"[{self.node_name}] Initiating sentiment analysis for text (length: {len(data)} characters).")

        # Simulate sentiment analysis with keyword matching
        text_lower = data.lower()
        
        positive_keywords = {
            "good", "great", "excellent", "happy", "love", "positive", "awesome",
            "fantastic", "superb", "wonderful", "amazing", "joy", "like"
        }
        negative_keywords = {
            "bad", "terrible", "horrible", "sad", "hate", "negative", "awful",
            "poor", "worse", "disappointing", "frustrated", "dislike"
        }

        positive_count = sum(text_lower.count(kw) for kw in positive_keywords)
        negative_count = sum(text_lower.count(kw) for kw in negative_keywords)

        sentiment = "Neutral"
        score = 0.0

        total_scoreable_keywords = positive_count + negative_count

        if total_scoreable_keywords > 0:
            if positive_count > negative_count:
                sentiment = "Positive"
                score = positive_count / total_scoreable_keywords
            elif negative_count > positive_count:
                sentiment = "Negative"
                score = - (negative_count / total_scoreable_keywords)
            else: # Equal counts of positive and negative keywords
                sentiment = "Neutral"
                score = 0.0
        else:
            # If no sentiment-bearing keywords found, default to neutral
            sentiment = "Neutral"
            score = 0.0

        logger.info(f"[{self.node_name}] Analysis concluded. Sentiment: '{sentiment}', Score: {score:.2f}.")

        return {
            "text": data,
            "sentiment": sentiment,
            "score": score
        }
