import logging
from typing import Any, Dict, Union

# Assuming this path for BaseNode as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on textual data.

    This node is designed to accept a string as input, analyze its sentiment
    (e.g., positive, negative, neutral), and return a structured dictionary
    containing the analysis results. The sentiment analysis is simulated
    using a simple keyword-based approach for demonstration purposes.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Analyzes the sentiment of the input text data.

        The method expects the input `data` to be a string. It simulates
        sentiment analysis and returns a dictionary with the text, sentiment
        label, and a confidence score.

        Args:
            data (Any): The input data, expected to be a string containing text
                        for sentiment analysis.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the node's operation. While not directly
                                       used in this simulated analysis, it's available
                                       for configuration (e.g., API keys, model names)
                                       in a real-world implementation.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the original text,
                                         the predicted sentiment ('positive', 'negative',
                                         'neutral'), and a confidence score.
                                         Example: {"text": "Hello world", "sentiment": "neutral", "score": 0.5}

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If an unexpected error occurs during the sentiment
                        analysis process.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for input of type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Input: {data!r}"
            )
            raise TypeError(
                f"{self.node_name} node requires string input for analysis, "
                f"but received {type(data).__name__}."
            )

        text_to_analyze = data.strip()
        if not text_to_analyze:
            logger.warning(
                f"[{self.node_name}] Received an empty or whitespace-only string for analysis. "
                f"Defaulting sentiment to 'neutral'."
            )
            return {
                "text": data,
                "sentiment": "neutral",
                "score": 0.5  # Default neutral score for empty input
            }

        try:
            # --- Sentiment Analysis Simulation ---
            # In a production system, this section would integrate with
            # a dedicated NLP library (e.g., NLTK, spaCy, TextBlob) or
            # an external sentiment analysis API (e.g., Hugging Face, AWS Comprehend,
            # Google Cloud Natural Language API).
            # For this demonstration, we use a simplistic keyword-based approach.

            lower_text = text_to_analyze.lower()
            sentiment = "neutral"
            score = 0.5

            positive_keywords = ["good", "great", "excellent", "happy", "love", "positive", "awesome", "fantastic", "amazing"]
            negative_keywords = ["bad", "terrible", "poor", "sad", "hate", "negative", "awful", "disappointing", "frustrating"]

            pos_count = sum(1 for keyword in positive_keywords if keyword in lower_text)
            neg_count = sum(1 for keyword in negative_keywords if keyword in lower_text)

            if pos_count > neg_count:
                sentiment = "positive"
                # Simple heuristic for score based on keyword count
                score = min(0.95, 0.5 + pos_count * 0.1)
            elif neg_count > pos_count:
                sentiment = "negative"
                # Simple heuristic for score based on keyword count
                score = max(0.05, 0.5 - neg_count * 0.1)
            else:
                sentiment = "neutral"
                score = 0.5 # Default score for neutral or mixed sentiment

            analysis_result = {
                "text": data,
                "sentiment": sentiment,
                "score": round(score, 4) # Rounding for consistent precision
            }

            logger.debug(
                f"[{self.node_name}] Successfully analyzed sentiment for text "
                f"'{text_to_analyze[:75]}...'. Result: {analysis_result}"
            )
            return analysis_result

        except Exception as e:
            # Catching generic exceptions for robustness, logging details
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during sentiment analysis "
                f"for text '{text_to_analyze[:75]}...'. Error: {e}"
            )
            raise ValueError(
                f"Failed to perform sentiment analysis in {self.node_name} "
                f"due to an internal processing error: {e}"
            ) from e
