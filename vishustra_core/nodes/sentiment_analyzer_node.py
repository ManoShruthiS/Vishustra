import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to perform sentiment analysis on input text.

    This implementation provides a basic, keyword-based simulation of sentiment
    classification (positive, negative, neutral). In a production scenario,
    this node would interface with a more sophisticated NLP model or external
    sentiment analysis service.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the provided text data.

        The node expects the input `data` to be a string containing the text
        to be analyzed. It classifies the sentiment as 'positive', 'negative',
        or 'neutral' based on a simple keyword matching heuristic.

        The `context` dictionary can optionally be used to override the default
        positive and negative keyword lists.

        Args:
            data: The text data (string) for which sentiment needs to be analyzed.
            context: A dictionary containing operational context or configuration.
                     Expected optional keys:
                       - 'positive_keywords': A `List[str]` of words indicating positive sentiment.
                       - 'negative_keywords': A `List[str]` of words indicating negative sentiment.

        Returns:
            A dictionary containing the original text and its classified sentiment.
            Example: {"text": "This product is great!", "sentiment": "positive"}

        Raises:
            ValueError: If the input `data` is not a string, as this node
                        specifically operates on textual input.
        """
        logger.info(f"'{self.node_name}' node initiated processing for incoming data.")

        if not isinstance(data, str):
            error_message = (
                f"'{self.node_name}' node received invalid input type. "
                f"Expected string, but got {type(data).__name__}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        text_lower = data.lower()
        sentiment = "neutral"

        # Define default keywords. These can be overridden via the context dictionary.
        default_positive_keywords: List[str] = ['great', 'excellent', 'love', 'happy', 'good', 'awesome', 'fantastic', 'superb']
        default_negative_keywords: List[str] = ['bad', 'terrible', 'hate', 'sad', 'poor', 'awful', 'dreadful', 'unpleasant']

        positive_keywords = context.get('positive_keywords', default_positive_keywords)
        negative_keywords = context.get('negative_keywords', default_negative_keywords)

        is_positive = any(kw in text_lower for kw in positive_keywords)
        is_negative = any(kw in text_lower for kw in negative_keywords)

        if is_positive and not is_negative:
            sentiment = "positive"
        elif is_negative and not is_positive:
            sentiment = "negative"
        elif is_positive and is_negative:
            # If both positive and negative keywords are found, it's ambiguous.
            # Defaulting to neutral or could be considered 'mixed' based on requirements.
            sentiment = "neutral"
            logger.debug(f"'{self.node_name}' node detected mixed sentiment for text (starts with: '{data[:50]}...'). Defaulting to neutral.")
        else:
            sentiment = "neutral"

        result = {"text": data, "sentiment": sentiment}
        logger.info(f"'{self.node_name}' node completed processing. Detected sentiment: '{sentiment}'.")
        return result