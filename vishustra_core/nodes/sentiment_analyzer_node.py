import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra node that performs sentiment analysis on input text.

    This node is designed to take a string as input, analyze its content,
    and return a structured dictionary containing the original text,
    the detected sentiment (e.g., 'positive', 'negative', 'neutral'),
    and a confidence score. For demonstration purposes, a simple
    keyword-based heuristic is used for sentiment detection.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The method expects `data` to be a string. It performs a basic
        keyword-based sentiment analysis and returns a dictionary with
        the results.

        Args:
            data: The input data, expected to be a string of text for analysis.
            context: A dictionary containing contextual information relevant
                     to the orchestration, which can be used for configuration
                     or state management (not explicitly used in this basic
                     implementation).

        Returns:
            A dictionary containing the analysis results:
                - "text": The original input text.
                - "sentiment": A string indicating "positive", "negative", or "neutral".
                - "score": A float representing the confidence score (0.0 to 1.0).

        Raises:
            TypeError: If the input data is not a string.
            Exception: For any other unexpected errors during the sentiment analysis process.
        """
        if not isinstance(data, str):
            error_message = (
                f"{self.node_name} received invalid data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(
                error_message,
                extra={"node_name": self.node_name, "input_type": type(data).__name__, "input_data_sample": str(data)[:100]}
            )
            raise TypeError(error_message)

        text_to_analyze = data.lower()
        sentiment = "neutral"
        score = 0.5  # Default neutral score

        try:
            # --- Simple heuristic for sentiment analysis simulation ---
            positive_keywords = ["good", "great", "excellent", "happy", "love", "joy", "positive", "awesome", "fantastic", "success"]
            negative_keywords = ["bad", "terrible", "horrible", "sad", "hate", "anger", "negative", "awful", "frustrating", "failure"]
            neutral_keywords = ["ok", "fine", "neutral", "average"]

            positive_count = sum(text_to_analyze.count(kw) for kw in positive_keywords)
            negative_count = sum(text_to_analyze.count(kw) for kw in negative_keywords)
            neutral_count = sum(text_to_analyze.count(kw) for kw in neutral_keywords)

            # Determine sentiment based on keyword counts
            if positive_count > negative_count and positive_count > neutral_count:
                sentiment = "positive"
                score = min(0.5 + (positive_count - negative_count) * 0.1, 0.98)
            elif negative_count > positive_count and negative_count > neutral_count:
                sentiment = "negative"
                score = max(0.5 - (negative_count - positive_count) * 0.1, 0.02)
            elif neutral_count > 0 and neutral_count >= positive_count and neutral_count >= negative_count:
                sentiment = "neutral"
                score = 0.5
            else:
                # If no strong bias, default to neutral
                sentiment = "neutral"
                score = 0.5 + (positive_count - negative_count) * 0.01 # Slightly adjust if some bias exists but not dominant

            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))

            result = {
                "text": data,  # Return original casing of the input text
                "sentiment": sentiment,
                "score": round(score, 3)
            }

            logger.debug(
                f"{self.node_name} processed text. Sentiment: {sentiment}, Score: {result['score']}",
                extra={"node_name": self.node_name, "input_text_snippet": data[:50], "analysis_result": result}
            )
            return result

        except Exception as e:
            error_message = f"An unexpected error occurred during sentiment analysis in {self.node_name}: {e}"
            logger.exception(
                error_message,
                extra={"node_name": self.node_name, "input_data": data}
            )
            raise # Re-raise the exception after logging to propagate the failure