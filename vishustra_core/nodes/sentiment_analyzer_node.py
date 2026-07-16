import logging
from typing import Any, Dict

# Assuming vishustra_core is installed and available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra processing node that performs basic sentiment analysis on text data.

    This node expects a string as input data and returns a dictionary
    containing the original text and its determined sentiment ('positive',
    'negative', or 'neutral') based on a simple keyword matching algorithm.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string) to determine its sentiment.

        The sentiment analysis is simulated based on simple keyword matching.
        The `context` parameter is available for more complex scenarios, such as
        accessing shared resources or configuration, but is not explicitly
        utilized in this basic implementation.

        Args:
            data: The input text data to analyze. Expected to be a string.
            context: A dictionary containing contextual information for the processing.

        Returns:
            A dictionary containing the original text and its determined sentiment.
            Example: {"text": "This is a great day!", "sentiment": "positive"}

        Raises:
            ValueError: If the input data is not a string.
            Exception: For unexpected errors during the sentiment analysis process.
        """
        node_id = context.get('node_id', self.node_name) # Use node_id from context if available
        logger.debug(f"[{node_id}] Starting sentiment analysis for data type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (
                f"[{node_id}] Invalid input data type for sentiment analysis. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_lower = data.lower()
        sentiment = "neutral"

        try:
            # Simple keyword-based sentiment detection for demonstration.
            # In a production environment, this would involve NLP libraries
            # or pre-trained machine learning models.
            positive_keywords = ["happy", "great", "excellent", "wonderful", "amazing", "love", "good", "perfect"]
            negative_keywords = ["sad", "bad", "terrible", "horrible", "awful", "hate", "poor", "unfortunate"]

            if any(keyword in text_lower for keyword in positive_keywords):
                sentiment = "positive"
            elif any(keyword in text_lower for keyword in negative_keywords):
                sentiment = "negative"
            
            # For a more nuanced approach, one might check for both and decide
            # on "mixed" or a score, but for this basic node, a clear category is preferred.

            result = {"text": data, "sentiment": sentiment}
            logger.debug(f"[{node_id}] Finished sentiment analysis. Detected sentiment: '{sentiment}'.")
            return result

        except Exception as e:
            error_msg = f"[{node_id}] An unexpected error occurred during sentiment analysis: {e}"
            logger.exception(error_msg) # Logs the exception traceback automatically
            raise # Re-raise the exception to propagate the error