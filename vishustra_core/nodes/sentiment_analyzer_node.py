import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs basic sentiment analysis on text data.
    This node identifies the overall sentiment (positive, negative, neutral) of a given
    text string using a simple keyword-based approach.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) to determine its sentiment.

        The analysis is based on a predefined set of positive and negative keywords.
        The `context` dictionary is provided for future extensibility but is not
        directly utilized in this basic implementation.

        Args:
            data: The input text string to analyze for sentiment.
            context: A dictionary containing contextual information relevant to the
                     orchestration flow (e.g., user preferences, system settings).

        Returns:
            A dictionary containing:
            - "text": The original input text.
            - "sentiment": A string indicating the detected sentiment ("positive",
                           "negative", or "neutral").
            - "score": A float representing a simplified sentiment score, typically
                       between -1.0 (very negative) and 1.0 (very positive).

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty or whitespace-only string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Input 'data' must be a string for sentiment "
                f"analysis, but received type: {type(data).__name__}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            error_msg = (
                f"[{self.node_name}] Received an empty or whitespace-only string "
                f"for sentiment analysis. Cannot process."
            )
            logger.warning(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"[{self.node_name}] Starting sentiment analysis for text snippet...")

        text_lower = data.lower()

        # Simple keyword-based sentiment detection for demonstration purposes.
        # In a production system, this would typically involve a sophisticated
        # NLP model (e.g., using a pre-trained transformer model or a specialized library).
        positive_keywords = ["good", "great", "excellent", "wonderful", "amazing",
                             "happy", "love", "fantastic", "superb", "positive",
                             "brilliant", "enjoy", "like", "recommend"]
        negative_keywords = ["bad", "terrible", "horrible", "awful", "hate", "poor",
                             "unpleasant", "fail", "disappointing", "negative",
                             "frustrating", "dislike", "avoid"]

        positive_count = sum(text_lower.count(kw) for kw in positive_keywords)
        negative_count = sum(text_lower.count(kw) for kw in negative_keywords)

        sentiment = "neutral"
        score = 0.0

        if positive_count > negative_count:
            sentiment = "positive"
            # A simplistic scoring mechanism: more positive words lead to higher score
            score = (positive_count - negative_count) / (positive_count + negative_count + 1e-9)
        elif negative_count > positive_count:
            sentiment = "negative"
            # A simplistic scoring mechanism: more negative words lead to lower score
            score = (positive_count - negative_count) / (positive_count + negative_count + 1e-9)
        else: # positive_count == negative_count or both are zero
            sentiment = "neutral"
            score = 0.0 # Explicitly zero for neutral

        result = {
            "text": data,
            "sentiment": sentiment,
            "score": round(score, 4) # Round for cleaner output
        }

        logger.debug(f"[{self.node_name}] Finished sentiment analysis. "
                     f"Detected sentiment: '{sentiment}' (Score: {result['score']:.2f})")

        return result