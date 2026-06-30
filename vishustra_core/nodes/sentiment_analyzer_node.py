import logging
from typing import Any, Dict

# Assuming vishustra_core is a package and nodes is a subpackage
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.

    This node simulates sentiment analysis by identifying positive, negative,
    or neutral sentiment based on a rudimentary keyword-matching approach.
    It expects string data as input and outputs a structured dictionary
    containing the analysis result.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzerNode"

    def _analyze_sentiment_simulated(self, text: str) -> Dict[str, Any]:
        """
        Simulates sentiment analysis for a given text using a keyword-based approach.

        This method is a placeholder for a more sophisticated sentiment analysis model
        and serves to demonstrate the node's functionality.

        Args:
            text: The input string to analyze.

        Returns:
            A dictionary containing the determined sentiment ('positive', 'negative', 'neutral'),
            a basic score, and details on keyword matches.
        """
        text_lower = text.lower()
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic", "wonderful", "amazing"]
        negative_keywords = ["bad", "terrible", "awful", "unhappy", "hate", "poor", "frustrating", "disappointing", "horrible"]

        positive_score = sum(text_lower.count(kw) for kw in positive_keywords)
        negative_score = sum(text_lower.count(kw) for kw in negative_keywords)

        sentiment: str
        if positive_score > negative_score:
            sentiment = "positive"
        elif negative_score > positive_score:
            sentiment = "negative"
        else:
            sentiment = "neutral" # Default if scores are equal or both zero

        return {
            "sentiment": sentiment,
            "compound_score": positive_score - negative_score, # A simple illustrative score
            "positive_keyword_matches": positive_score,
            "negative_keyword_matches": negative_score,
            "analysis_method": "simulated_keyword_analysis"
        }

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, performing sentiment analysis if the data is a string.

        Args:
            data: The input data, expected to be a string containing text for analysis.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context for its core logic,
                     but it's available for future enhancements (e.g., model configuration).

        Returns:
            A dictionary containing:
            - 'original_text': The input text.
            - 'analysis_result': A nested dictionary with sentiment, score, and method details.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during the sentiment analysis process.
        """
        logger.info(f"[{self.node_name}] Initiating sentiment analysis process.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type for sentiment analysis. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            analysis_result = self._analyze_sentiment_simulated(data)
            output = {
                "original_text": data,
                "analysis_result": analysis_result
            }
            logger.debug(f"[{self.node_name}] Successfully completed sentiment analysis. Sentiment: '{analysis_result['sentiment']}'.")
            return output
        except Exception as e:
            # Use logger.exception to include traceback in the log
            error_msg = f"[{self.node_name}] An unexpected error occurred during sentiment analysis: {e}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e