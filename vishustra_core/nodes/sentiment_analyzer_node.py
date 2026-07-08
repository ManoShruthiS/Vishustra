import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on text data.

    This node expects the input 'data' to be a string. It performs a simplified
    sentiment analysis, classifying the input as positive, negative, mixed,
    or neutral based on keyword matching. The node is designed to integrate
    seamlessly into an orchestration flow where textual data requires
    preliminary sentiment assessment.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input text data to determine its sentiment.

        The method expects 'data' to be a string containing the text to be
        analyzed. It identifies a sentiment (positive, negative, mixed, neutral)
        and provides a simulated confidence score.

        Args:
            data: The input data to process. Expected to be a string representing
                  the text for sentiment analysis.
            context: A dictionary containing additional runtime context,
                     which might include configuration settings or shared state
                     relevant to the current orchestration run.

        Returns:
            A dictionary containing the original text, the determined sentiment,
            and a simulated confidence score.
            Example: {"text": "This movie was absolutely fantastic!", "sentiment": "positive", "score": 0.95}

        Raises:
            TypeError: If the input 'data' is not of type string, ensuring
                       data integrity and proper type adherence within the pipeline.
        """
        if not isinstance(data, str):
            error_message = (
                f"SentimentAnalyzerNode received invalid data type. "
                f"Expected 'str', but got '{type(data).__name__}'."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        text_lower = data.strip().lower()
        sentiment = "neutral"
        score = 0.5  # Default neutral score

        if not text_lower:
            logger.info("Received an empty string for sentiment analysis. Returning neutral sentiment.")
            return {"text": data, "sentiment": sentiment, "score": round(score, 2)}

        # Simplified keyword-based sentiment detection
        positive_keywords = {"good", "great", "excellent", "happy", "love", "awesome", "fantastic", "amazing", "superb"}
        negative_keywords = {"bad", "terrible", "poor", "sad", "hate", "awful", "horrible", "frustrating", "disappointing"}

        is_positive = any(keyword in text_lower for keyword in positive_keywords)
        is_negative = any(keyword in text_lower for keyword in negative_keywords)

        if is_positive and not is_negative:
            sentiment = "positive"
            # Simulate a higher score for more emphasis
            score = 0.8 + (text_lower.count("!") * 0.05)
            score = min(score, 0.99) # Cap at near max
            logger.debug(f"Detected positive sentiment for text: '{data}'")
        elif is_negative and not is_positive:
            sentiment = "negative"
            # Simulate a lower score for more emphasis
            score = 0.2 - (text_lower.count("!") * 0.05)
            score = max(score, 0.01) # Cap at near min
            logger.debug(f"Detected negative sentiment for text: '{data}'")
        elif is_positive and is_negative:
            sentiment = "mixed"
            score = 0.5
            logger.debug(f"Detected mixed (positive and negative) keywords for text: '{data}'")
        else:
            sentiment = "neutral"
            score = 0.5
            logger.debug(f"No strong sentiment keywords detected for text: '{data}'")

        result = {
            "text": data, # Preserve original casing of the input text
            "sentiment": sentiment,
            "score": round(score, 2) # Round score for consistent output
        }
        logger.debug(f"Processed text: '{data}' -> Result: {result}")
        return result
