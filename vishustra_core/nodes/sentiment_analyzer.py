import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra node that performs a simulated sentiment analysis on text data.
    This node categorizes input text as 'positive', 'negative', or 'neutral'
    based on simple keyword matching. It's designed to illustrate data transformation
    within the Vishustra framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        This method expects `data` to be a string containing the text to be analyzed.
        It returns a dictionary with the categorized sentiment and a simulated score.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This can be
                                       used for logging additional details or dynamic
                                       configuration if needed.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis result, e.g.,
                            `{'sentiment': 'positive', 'score': 0.85}`. The 'score'
                            is a float between -1.0 (highly negative) and 1.0
                            (highly positive).

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an unexpected issue occurs during the sentiment analysis
                        simulation.
        """
        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Input data must be a string for sentiment analysis, "
                f"but received type: {type(data).__name__}. Data: {repr(data)[:100]}"
            )
            logger.error(error_message, exc_info=True)
            raise TypeError(error_message)

        text_to_analyze = data.lower()
        sentiment = "neutral"
        score = 0.0

        logger.debug(
            f"[{self.node_name}] Starting sentiment analysis for text (truncated): "
            f"'{text_to_analyze[:75]}{'...' if len(text_to_analyze) > 75 else ''}'"
        )

        try:
            # Simple keyword-based sentiment simulation for demonstration purposes.
            # In a real-world scenario, this would involve a robust NLP model or API call.
            positive_keywords = {"good", "great", "excellent", "happy", "love", "awesome", "fantastic", "amazing", "superb"}
            negative_keywords = {"bad", "terrible", "poor", "sad", "hate", "awful", "horrible", "disappointing", "frustrating"}

            is_positive = any(keyword in text_to_analyze for keyword in positive_keywords)
            is_negative = any(keyword in text_to_analyze for keyword in negative_keywords)

            if is_positive and not is_negative:
                sentiment = "positive"
                score = 0.7 + (text_to_analyze.count("!") * 0.05)  # Boost score for exclamation marks
                score = min(score, 0.95)  # Cap positive score
            elif is_negative and not is_positive:
                sentiment = "negative"
                score = -0.7 - (text_to_analyze.count("!") * 0.05) # Depress score for exclamation marks
                score = max(score, -0.95) # Cap negative score
            elif is_positive and is_negative:
                # If both positive and negative keywords are present, it's ambiguous.
                # A more sophisticated model would resolve this. For simulation, default to neutral
                # or a very mild score based on presence.
                sentiment = "neutral"
                score = 0.0
            else:
                sentiment = "neutral"
                score = 0.1 # A slight positive bias for general unknown text, or could be 0.0

            # Format score to two decimal places for consistent output
            result = {"sentiment": sentiment, "score": float(f"{score:.2f}")}

            logger.info(f"[{self.node_name}] Analysis complete. Result: {result}")
            return result
        except Exception as e:
            critical_error_message = (
                f"[{self.node_name}] An unexpected error occurred during sentiment "
                f"analysis simulation for data: '{text_to_analyze[:75]}...': {e}"
            )
            logger.critical(critical_error_message, exc_info=True)
            raise ValueError(critical_error_message) from e
