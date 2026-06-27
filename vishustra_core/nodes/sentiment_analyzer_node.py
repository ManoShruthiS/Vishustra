from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, Literal
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on input text.
    It categorizes text as 'positive', 'negative', or 'neutral' based on simple
    keyword matching.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data, which is expected to be a string.

        The node assigns a sentiment label ('positive', 'negative', 'neutral')
        based on a predefined set of keywords. If the input is not a string,
        it logs an error and raises a TypeError.

        Args:
            data: The input text (string) to be analyzed for sentiment.
            context: A dictionary containing contextual information for processing.
                     (Not directly used in this basic sentiment analysis, but passed along).

        Returns:
            A dictionary containing:
            - 'original_text': The input text.
            - 'sentiment': The detected sentiment ('positive', 'negative', 'neutral').
            - 'node_name': The name of this processing node.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"SentimentAnalyzerNode expects string input, but received type: {type(data).__name__}. Data: {data!r}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_lower = data.lower()
        sentiment: Literal['positive', 'negative', 'neutral'] = 'neutral'

        positive_keywords = ['good', 'great', 'excellent', 'happy', 'love', 'amazing', 'fantastic', 'awesome', 'wonderful', 'joy']
        negative_keywords = ['bad', 'terrible', 'horrible', 'sad', 'hate', 'awful', 'poor', 'disappointed', 'ugly', 'fail']

        is_positive = any(word in text_lower for word in positive_keywords)
        is_negative = any(word in text_lower for word in negative_keywords)

        if is_negative and not is_positive:
            sentiment = 'negative'
        elif is_positive and not is_negative:
            sentiment = 'positive'
        elif is_positive and is_negative:
            # If both positive and negative words are present,
            # a more sophisticated model would be needed.
            # For this simple simulation, we'll lean towards neutral/mixed.
            sentiment = 'neutral'
        elif not text_lower.strip():
            # Handle empty or whitespace-only strings
            sentiment = 'neutral'

        logger.info(f"Analyzed sentiment for text (first 50 chars): '{data[:50]}...' -> {sentiment}")

        return {
            "original_text": data,
            "sentiment": sentiment,
            "node_name": self.node_name
        }