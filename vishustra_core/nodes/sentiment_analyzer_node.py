import logging
from typing import Any, Dict

# Assuming vishustra_core is installed and available in the project environment.
# The BaseNode class definition provided in the prompt context implies this import path.
from vishustra_core.nodes.base_node import BaseNode # type: ignore

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to analyze the sentiment of input text.

    This node simulates sentiment analysis by identifying predefined positive and
    negative keywords within the text. It's intended to be a robust, modular
    component within the Vishustra orchestration framework.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data expected to be a string containing text
                        for sentiment analysis.
            context (Dict[str, Any]): A dictionary containing runtime context
                                      information. Not directly used by this
                                      node's core logic but available for
                                      future extensions or orchestration hints.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'text': The original input text.
                - 'sentiment': The identified sentiment ('positive', 'negative', 'neutral').
                - 'score': A float representing the strength of the sentiment (e.g.,
                           positive for positive, negative for negative, 0 for neutral).

        Raises:
            ValueError: If the input 'data' is not a string or is an empty string.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for data.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected a string, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_to_analyze = data.strip()
        if not text_to_analyze:
            error_msg = f"[{self.node_name}] Input text is empty. Cannot perform sentiment analysis on empty data."
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # Simple keyword-based sentiment detection for simulation purposes.
        # In a production system, this would involve a robust NLP library or ML model.
        positive_keywords = {
            "good", "great", "excellent", "happy", "love", "awesome",
            "fantastic", "wonderful", "positive", "superb", "brilliant",
            "joy", "pleasure", "amazing", "thrilled"
        }
        negative_keywords = {
            "bad", "terrible", "horrible", "sad", "hate", "unhappy",
            "awful", "negative", "dreadful", "poor", "pain", "failure",
            "disappointing", "frustrating"
        }

        words = text_to_analyze.lower().split()

        positive_count = sum(1 for word in words if word in positive_keywords)
        negative_count = sum(1 for word in words if word in negative_keywords)

        sentiment: str = "neutral"
        score: float = 0.0

        if positive_count > negative_count:
            sentiment = "positive"
            # Score normalized by total relevant words for a simple representation
            total_relevant_words = positive_count + negative_count
            score = positive_count / total_relevant_words if total_relevant_words > 0 else 0.0
        elif negative_count > positive_count:
            sentiment = "negative"
            total_relevant_words = positive_count + negative_count
            score = -negative_count / total_relevant_words if total_relevant_words > 0 else 0.0
        else:
            # If counts are equal, or both are zero
            sentiment = "neutral"
            score = 0.0

        # Format the result consistently
        result = {
            "text": data,
            "sentiment": sentiment,
            "score": round(score, 4)
        }

        logger.info(f"[{self.node_name}] Sentiment analysis concluded. Result: {result['sentiment']} (score: {result['score']}).")
        return result