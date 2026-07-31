import logging
from typing import Any, Dict

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class VishustraNodeError(Exception):
    """Custom exception for errors specific to Vishustra processing nodes."""
    pass

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node designed to perform sentiment analysis on input text.

    This node accepts a string as input, analyzes its sentiment (categorizing it
    as positive, negative, or neutral) using a simplified keyword-based approach,
    and returns a dictionary containing the original text and the determined sentiment.
    It's intended for initial data enrichment or as part of a larger NLP pipeline.
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

        The method expects `data` to be a string. It will analyze the text
        for predefined positive and negative keywords to assign a sentiment.

        Args:
            data: The input data, expected to be a string containing the text
                  for which sentiment analysis is required.
            context: A dictionary containing contextual information, potentially
                     including a 'node_id' for logging purposes.

        Returns:
            A dictionary structured as `{'text': original_text, 'sentiment': 'category'}`,
            where 'category' can be 'positive', 'negative', or 'neutral'.

        Raises:
            VishustraNodeError: If the input `data` is not a string, indicating
                                an invalid input type for this node's operation.
        """
        # Retrieve a unique identifier for this node instance from context,
        # falling back to the generic node name if not provided.
        node_id_in_context = context.get('node_id', self.node_name)
        logger.info(f"[{node_id_in_context}] Starting sentiment analysis for data type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (
                f"[{node_id_in_context}] Invalid input data type for sentiment analysis. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise VishustraNodeError(error_msg)

        text = data.strip()
        if not text:
            logger.warning(f"[{node_id_in_context}] Received an empty string for analysis. Defaulting sentiment to 'neutral'.")
            return {"text": data, "sentiment": "neutral"}

        # Define keyword sets for simplified sentiment detection.
        # In a real-world scenario, this would involve a more sophisticated NLP model.
        positive_keywords = {
            "good", "great", "excellent", "happy", "love", "awesome",
            "fantastic", "amazing", "wonderful", "joyful", "positive", "superb"
        }
        negative_keywords = {
            "bad", "terrible", "poor", "sad", "hate", "awful",
            "horrible", "frustrating", "disappointing", "angry", "negative", "dreadful"
        }

        # Convert text to lowercase and split into words for keyword matching
        words = set(text.lower().split())

        positive_matches = sum(1 for word in words if word in positive_keywords)
        negative_matches = sum(1 for word in words if word in negative_keywords)

        sentiment = "neutral"
        if positive_matches > negative_matches:
            sentiment = "positive"
        elif negative_matches > positive_matches:
            sentiment = "negative"
        # If counts are equal, or both are zero, sentiment remains "neutral"

        result = {"text": data, "sentiment": sentiment}
        logger.debug(f"[{node_id_in_context}] Sentiment analysis completed. Result: {result}")
        return result