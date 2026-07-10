import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to perform sentiment analysis on input text data.

    This node simulates sentiment analysis by identifying a predefined set of
    keywords within the input string. It categorizes the text's sentiment as
    positive, negative, or neutral and provides a corresponding numerical score.
    In a production environment, this node would typically integrate with
    a dedicated natural language processing (NLP) library or an external
    large language model (LLM) service to determine sentiment more robustly.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "Sentiment Analyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        Expected `data` type: `str`.
        The `context` dictionary is provided for potential future extensions,
        allowing for configuration or additional runtime information, though
        it is not directly utilized in this current simulated version.

        Args:
            data (Any): The input data, expected to be a string
                        containing the text to be analyzed.
            context (Dict[str, Any]): A dictionary containing runtime
                                      contextual information.

        Returns:
            Dict[str, Any]: A dictionary containing the processed text,
                            the determined sentiment ('positive', 'negative', 'neutral'),
                            and a numerical sentiment score (typically between -1.0 and 1.0).

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Attempting sentiment analysis for data of type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_input = data.strip()
        if not text_input:
            error_msg = f"[{self.node_name}] Input data is an empty string after stripping. Cannot perform sentiment analysis."
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # Simulate sentiment analysis using a simple keyword-based approach.
        # This section would be replaced by an actual NLP library call or LLM API integration.
        lower_text = text_input.lower()
        sentiment = "neutral"
        score = 0.0

        positive_keywords = ["great", "excellent", "love", "happy", "wonderful", "amazing", "good", "fantastic", "positive"]
        negative_keywords = ["bad", "terrible", "hate", "unhappy", "awful", "poor", "disappointing", "negative", "horrible"]

        positive_count = sum(1 for keyword in positive_keywords if keyword in lower_text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in lower_text)

        if positive_count > negative_count:
            sentiment = "positive"
            # Simple scoring: more positive keywords, higher score. Max 1.0.
            score = min(1.0, 0.5 + (0.1 * positive_count))
        elif negative_count > positive_count:
            sentiment = "negative"
            # Simple scoring: more negative keywords, lower score. Min -1.0.
            score = max(-1.0, -0.5 - (0.1 * negative_count))
        else:
            sentiment = "neutral"
            score = 0.0 # Default neutral score

        # Ensure score is clamped within the expected range [-1.0, 1.0]
        score = max(-1.0, min(1.0, score))

        result = {
            "text": text_input,
            "sentiment": sentiment,
            "score": float(f"{score:.2f}") # Format score to two decimal places for cleaner output
        }

        log_text_preview = text_input if len(text_input) <= 70 else f"{text_input[:67]}..."
        logger.info(f"[{self.node_name}] Analyzed text '{log_text_preview}' -> Sentiment: {sentiment}, Score: {result['score']:.2f}")
        return result
