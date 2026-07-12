import logging
from typing import Any, Dict

# Importing BaseNode from the specified project path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that analyzes the sentiment of a given text.

    This node simulates sentiment analysis by identifying simple positive and
    negative keywords within the input data. In a production environment,
    this would typically integrate with a dedicated NLP service or model.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        The `data` input is expected to be a string. The node will classify
        the sentiment as 'Positive', 'Negative', or 'Neutral' based on
        predefined keywords.

        Args:
            data (Any): The input data, expected to be a string containing text
                        for sentiment analysis.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information for the processing pipeline.
                                      This implementation does not directly use
                                      the context for sentiment logic, but it is
                                      available for future extensions (e.g.,
                                      custom keyword lists from context).

        Returns:
            Dict[str, Any]: A dictionary containing the original text and its
                            determined sentiment.
                            Example: {"text": "This is great!", "sentiment": "Positive"}

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for SentimentAnalyzerNode. "
                f"Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"SentimentAnalyzerNode expects string input for analysis, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(
                "Received empty or whitespace-only string for sentiment analysis. "
                "Classifying as 'Neutral'."
            )
            return {"text": data, "sentiment": "Neutral"}

        text_lower = data.lower()

        # Simple keyword-based sentiment detection
        positive_keywords = [
            "good", "great", "excellent", "happy", "love", "joy",
            "fantastic", "amazing", "wonderful", "brilliant", "positive"
        ]
        negative_keywords = [
            "bad", "terrible", "poor", "sad", "hate", "anger",
            "awful", "horrible", "frustrating", "difficult", "negative"
        ]

        positive_score = sum(text_lower.count(kw) for kw in positive_keywords)
        negative_score = sum(text_lower.count(kw) for kw in negative_keywords)

        sentiment = "Neutral"
        if positive_score > negative_score:
            sentiment = "Positive"
        elif negative_score > positive_score:
            sentiment = "Negative"

        logger.debug(
            f"Analyzed text (first 50 chars): '{data[:50]}...' "
            f"Scores: P={positive_score}, N={negative_score} - Sentiment: {sentiment}"
        )
        return {"text": data, "sentiment": sentiment}
