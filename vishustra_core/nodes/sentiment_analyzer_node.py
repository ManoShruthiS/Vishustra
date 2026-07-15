import logging
from typing import Any, Dict

# Assuming BaseNode is located as per the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on text data.

    This node is designed to accept a string as input, representing a piece of text,
    and simulate the process of determining its sentiment (positive, negative, or neutral).
    It returns a structured dictionary containing the original text, the assigned
    sentiment label, and a simulated confidence score.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name for this processing node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its simulated sentiment.

        The method expects a string as input `data`. It performs a basic
        keyword-based analysis to assign a sentiment label and a confidence score.
        If the input is not a string, a ValueError is raised.

        Args:
            data: The input text data to be analyzed for sentiment. Expected type is `str`.
            context: A dictionary containing runtime context information.
                     (Currently not utilized by this specific node, but available for future extensions.)

        Returns:
            A dictionary containing the sentiment analysis results:
            - 'text': The original input string.
            - 'sentiment': A string indicating the detected sentiment ('positive', 'negative', 'neutral').
            - 'score': A float representing the simulated confidence score for the sentiment (0.0 to 1.0).

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid input type.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for input data type: {type(data).__name__}")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type received. Expected 'str', but got '{type(data).__name__}'.")
            raise ValueError(f"SentimentAnalyzerNode expects string input for analysis, but received type '{type(data).__name__}'.")

        text_lower = data.lower()
        sentiment = "neutral"
        score = 0.5  # Default neutral score

        # Simple keyword-based simulation for sentiment detection
        positive_keywords = ["good", "great", "excellent", "amazing", "happy", "love", "awesome", "fantastic"]
        negative_keywords = ["bad", "terrible", "poor", "awful", "unhappy", "hate", "disappointing", "worst"]

        if any(keyword in text_lower for keyword in positive_keywords):
            sentiment = "positive"
            score = 0.85
        elif any(keyword in text_lower for keyword in negative_keywords):
            sentiment = "negative"
            score = 0.90
        else:
            # Assign a slightly higher neutral score if no strong sentiment keywords are found
            sentiment = "neutral"
            score = 0.65

        result = {
            "text": data,  # Retain original casing of the input text
            "sentiment": sentiment,
            "score": score,
        }

        logger.info(f"[{self.node_name}] Sentiment analysis complete for text (truncated): '{data[:50]}{'...' if len(data) > 50 else ''}'. Result: {sentiment} (Score: {score:.2f})")
        return result