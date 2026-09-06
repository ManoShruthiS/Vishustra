
import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is discoverable in the project
# For this example, we'll use a relative import or an explicit one if the project structure supports it.
# If base_node is in a sibling directory, it might look like:
# from ..base_node import BaseNode
# Given the prompt, it's likely intended as a direct import path, implying it's part of the installed Vishustra package or available in the Python path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.

    This node simulates sentiment analysis by identifying positive and negative keywords
    within the input data. In a production environment, this would integrate with
    a robust NLP library or a dedicated sentiment analysis model.
    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzerNode.

        Sets up keyword lists for simulating sentiment detection.
        """
        self._positive_keywords = [
            "good", "great", "excellent", "love", "happy", "awesome",
            "fantastic", "brilliant", "positive", "superb", "enjoy"
        ]
        self._negative_keywords = [
            "bad", "terrible", "awful", "hate", "unhappy", "poor",
            "frustrating", "negative", "disappointing", "issue", "problem"
        ]
        logger.debug("SentimentAnalyzerNode initialized with keyword lists.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "Sentiment Analyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data to analyze. Expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            Dict[str, Any]: A dictionary containing the original text, the detected
                            sentiment ("positive", "negative", or "neutral"),
                            and a simulated sentiment score (from -1.0 to 1.0).

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `data` is an empty string after stripping whitespace.
        """
        logger.info(f"[{self.node_name}] Starting sentiment analysis for data type: {type(data).__name__}")
        logger.debug(f"[{self.node_name}] Context received: {context}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(f"[{self.node_name}] Input data is empty or only whitespace. Assigning neutral sentiment.")
            return {"text": data, "sentiment": "neutral", "score": 0.0}

        text_lower = stripped_data.lower()
        sentiment = "neutral"
        score = 0.0

        # Count keyword occurrences
        num_positive_matches = sum(1 for keyword in self._positive_keywords if keyword in text_lower)
        num_negative_matches = sum(1 for keyword in self._negative_keywords if keyword in text_lower)

        if num_positive_matches > num_negative_matches:
            sentiment = "positive"
            # Simple scoring: more matches -> higher score, capped at 1.0
            score = min(1.0, num_positive_matches * 0.2)
        elif num_negative_matches > num_positive_matches:
            sentiment = "negative"
            # Simple scoring: more matches -> lower score, floored at -1.0
            score = max(-1.0, num_negative_matches * -0.2)
        else:
            # If counts are equal or both zero, it's neutral.
            # We can introduce a slight bias towards neutral if no strong signals.
            sentiment = "neutral"
            score = 0.0 # Could add a tiny random offset like random.uniform(-0.05, 0.05) for more realism

        result = {"text": data, "sentiment": sentiment, "score": round(score, 2)}
        logger.info(f"[{self.node_name}] Analyzed text (first 50 chars): '{stripped_data[:50]}...' -> Result: {result}")

        return result

