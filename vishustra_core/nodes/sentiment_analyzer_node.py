from typing import Any, Dict
import logging

# Assuming BaseNode is available at the specified path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that analyzes the sentiment of input text.

    This node expects the input `data` to be either a string containing the text
    to analyze or a dictionary that includes a 'text' key with the relevant string value.
    It simulates sentiment analysis and returns a structured dictionary
    with the detected sentiment and a corresponding score.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data, expected to be a string or a dict
                        containing a 'text' key.
            context (Dict[str, Any]): A dictionary containing execution context
                                       information (unused in this specific node,
                                       but part of the BaseNode interface).

        Returns:
            Dict[str, Any]: A dictionary containing the original input, the
                            extracted text, and the sentiment analysis result
                            (sentiment label and a numeric score).
                            If processing fails, an 'error' key will be present.
        """
        extracted_text: str = ""
        analysis_result: Dict[str, Any] = {
            "original_input": data,
            "text": None,
            "sentiment": "neutral",
            "score": 0.0,
        }

        if isinstance(data, str):
            extracted_text = data
        elif isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
            extracted_text = data["text"]
        else:
            error_msg = (
                f"Invalid input data type for SentimentAnalyzerNode. "
                f"Expected str or dict with 'text' key, got {type(data)}."
            )
            logger.error(error_msg, extra={"node": self.node_name, "data_type": type(data)})
            analysis_result["error"] = error_msg
            return analysis_result

        extracted_text_lower = extracted_text.lower()
        analysis_result["text"] = extracted_text

        # Simulate sentiment analysis
        # In a real-world scenario, this would involve calling an NLP model
        # or a dedicated sentiment analysis service.
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic"]
        negative_keywords = ["bad", "terrible", "horrible", "sad", "hate", "awful", "unfortunate"]

        score = 0.0
        sentiment_label = "neutral"

        # Simple keyword-based sentiment detection
        for keyword in positive_keywords:
            if keyword in extracted_text_lower:
                score += 1.0
        for keyword in negative_keywords:
            if keyword in extracted_text_lower:
                score -= 1.0

        if score > 0:
            sentiment_label = "positive"
        elif score < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        analysis_result["sentiment"] = sentiment_label
        analysis_result["score"] = score

        logger.debug(
            f"Sentiment analysis complete for text: '{extracted_text[:50]}...'",
            extra={"node": self.node_name, "sentiment": sentiment_label, "score": score}
        )
        return analysis_result
