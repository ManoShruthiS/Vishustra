import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node designed to perform sentiment analysis on input text.

    This node processes a given string, analyzes its content, and determines
    a sentiment label ('positive', 'negative', 'neutral'). It's intended
    to be a flexible component in an orchestration pipeline, allowing for
    further processing based on sentiment.

    The current implementation uses a basic keyword-matching approach for
    demonstration purposes. In a production environment, this would typically
    interface with a sophisticated NLP model or a dedicated sentiment analysis service.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the input data (expected to be a string) to determine its sentiment.

        Args:
            data: The input text content to be analyzed. Must be a string.
            context: A dictionary containing execution context information.
                     This can include configuration, session data, or other
                     metadata relevant to the current pipeline run.
                     (Not directly used in this basic sentiment logic, but available).

        Returns:
            A dictionary containing:
            - 'text': The original input text.
            - 'sentiment': The determined sentiment ('positive', 'negative', 'neutral').

        Raises:
            TypeError: If the input `data` is not a string, indicating an invalid
                       input type for sentiment analysis.
        """
        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        text_lower = data.lower().strip()
        sentiment = "neutral"

        if not text_lower:
            logger.info(
                "[%s] Received empty string for analysis. Defaulting to 'neutral' sentiment.",
                self.node_name
            )
            # An empty string carries no sentiment.
            sentiment = "neutral"
        else:
            # Simple keyword-based sentiment detection for demonstration.
            # This logic is intentionally basic and would be replaced by
            # integration with an ML model or service in a real application.
            positive_keywords = ["great", "excellent", "love", "fantastic", "happy", "good", "amazing", "superb"]
            negative_keywords = ["bad", "terrible", "hate", "awful", "unhappy", "poor", "disappointing"]

            is_positive = any(keyword in text_lower for keyword in positive_keywords)
            is_negative = any(keyword in text_lower for keyword in negative_keywords)

            if is_positive and not is_negative:
                sentiment = "positive"
            elif is_negative and not is_positive:
                sentiment = "negative"
            elif is_positive and is_negative:
                # If both positive and negative indicators are present,
                # we consider it mixed or complex, defaulting to neutral for simplicity.
                logger.warning(
                    "[%s] Detected both positive and negative indicators in text: '%s'. "
                    "Defaulting to 'neutral' due to mixed signals.",
                    self.node_name,
                    data[:100] + "..." if len(data) > 100 else data
                )
                sentiment = "neutral"
            else:
                # No strong indicators found
                sentiment = "neutral"

        logger.info(
            "[%s] Analyzed text (truncated): '%s' -> Detected sentiment: '%s'",
            self.node_name,
            data[:100] + "..." if len(data) > 100 else data,
            sentiment
        )

        return {"text": data, "sentiment": sentiment}
