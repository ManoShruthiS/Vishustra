import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.
    It identifies whether the sentiment of the text is positive, negative, or neutral.
    
    This implementation uses a simple keyword-based heuristic for demonstration.
    In a production environment, this would typically integrate with an NLP library
    or a dedicated sentiment analysis service.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data, expected to be a string containing text
                        for sentiment analysis.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the node's operation.

        Returns:
            Dict[str, Any]: A dictionary containing the original text, the detected
                            sentiment ("positive", "negative", "neutral"), and a
                            corresponding sentiment score.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        logger.debug("SentimentAnalyzerNode received data for processing.")

        if not isinstance(data, str):
            error_msg = (
                f"SentimentAnalyzerNode expects string input for sentiment analysis, "
                f"but received type: {type(data).__name__}. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        text_input = data.strip()
        if not text_input:
            logger.warning("SentimentAnalyzerNode received an empty or whitespace-only string. Returning neutral sentiment.")
            return {"text": data, "sentiment": "neutral", "score": 0.0, "reason": "empty_input"}

        sentiment = "neutral"
        score = 0.0

        # Simple keyword-based heuristic for sentiment analysis simulation
        text_lower = text_input.lower()
        
        positive_keywords = ["good", "great", "excellent", "positive", "happy", "love", "awesome", "fantastic", "amazing"]
        negative_keywords = ["bad", "terrible", "horrible", "negative", "sad", "hate", "awful", "poor", "unpleasant"]

        if any(word in text_lower for word in positive_keywords):
            sentiment = "positive"
            score = 0.85  # Arbitrary positive score
        elif any(word in text_lower for word in negative_keywords):
            sentiment = "negative"
            score = -0.85 # Arbitrary negative score
        else:
            sentiment = "neutral"
            score = 0.0

        result = {
            "text": data,
            "sentiment": sentiment,
            "score": score
        }
        
        log_message_text = data[:75] + "..." if len(data) > 75 else data
        logger.info(
            f"Sentiment analyzed for text: '{log_message_text}' -> "
            f"Sentiment: '{sentiment}', Score: {score:.2f}"
        )
        return result