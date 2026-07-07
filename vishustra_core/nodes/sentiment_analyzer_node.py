import logging
from typing import Any, Dict

# Assuming BaseNode is available in the specified module path as per project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that analyzes the sentiment of input text.

    This node simulates sentiment analysis, categorizing text as 'positive',
    'negative', or 'neutral' based on simplified keyword matching.
    In a production environment, this would typically integrate with an actual NLP
    sentiment analysis service, a pre-trained model, or an external API.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.

        Expects 'data' to be a string. It returns a dictionary containing
        the original text and its inferred sentiment.

        Args:
            data (Any): The input data, expected to be a string of text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing. This could include
                                       configuration settings for a more complex analyzer.

        Returns:
            Dict[str, Any]: A dictionary with 'text' and 'sentiment' keys.
                            Example: {"text": "Hello world!", "sentiment": "neutral"}
                            An additional 'message' key might be present for edge cases.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name} received non-string data. "
                f"Type: {type(data).__name__}. Expected 'str'."
            )
            logger.error(error_msg, extra={"data_type": type(data).__name__})
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning(
                f"{self.node_name} received an empty string for analysis. "
                "Returning 'neutral' sentiment.",
                extra={"original_data": data}
            )
            return {"text": data, "sentiment": "neutral", "message": "Empty string provided."}

        text_lower = data.lower()
        sentiment = "neutral"

        # Simplified keyword-based sentiment detection for demonstration.
        # This would be replaced by actual NLP model inference in a real system.
        positive_keywords = ["good", "happy", "great", "excellent", "love", "joy", "positive"]
        negative_keywords = ["bad", "sad", "terrible", "awful", "hate", "unhappy", "negative"]

        if any(keyword in text_lower for keyword in positive_keywords):
            sentiment = "positive"
        elif any(keyword in text_lower for keyword in negative_keywords):
            sentiment = "negative"

        result = {"text": data, "sentiment": sentiment}
        logger.debug(
            f"{self.node_name} processed text (first 50 chars): '{data[:50]}...' "
            f"- Inferred sentiment: '{sentiment}'",
            extra={"sentiment_result": sentiment, "original_text_preview": data[:50]}
        )

        return result
