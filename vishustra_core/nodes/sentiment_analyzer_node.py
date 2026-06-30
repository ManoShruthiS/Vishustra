import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node designed to perform sentiment analysis on input text data.

    This node simulates sentiment analysis by identifying key positive and negative
    terms within the provided text. It classifies the sentiment as positive,
    negative, neutral, or mixed, and provides a simulated confidence score.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.

        The `data` parameter is expected to be a string containing the text
        to be analyzed. The method returns a dictionary detailing the detected
        sentiment, a simulated confidence score, and any relevant processing
        details. Robust error handling is included to manage invalid input types.

        Args:
            data: The input text (str) to analyze for sentiment.
            context: A dictionary containing contextual information. Not directly
                     used by this node for processing logic but available for
                     future extensions (e.g., custom keyword lists).

        Returns:
            A dictionary containing the sentiment analysis results:
            - 'sentiment': 'positive', 'negative', 'neutral', 'mixed'
            - 'score': A float between 0.0 and 1.0 representing confidence.
            - 'node_name': The name of this node.
            - 'details': A string providing additional information about the analysis.
            If an error occurs (e.g., invalid input type), an error dictionary
            is returned with 'error': True and a 'message'.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            return {
                "error": True,
                "message": error_msg,
                "node_name": self.node_name,
                "input_type": type(data).__name__
            }

        text_lower = data.lower()
        sentiment = "neutral"
        score = 0.5
        analysis_details = []

        positive_keywords = {"great", "excellent", "happy", "love", "awesome", "fantastic", "good", "wonderful", "amazing", "superb"}
        negative_keywords = {"bad", "terrible", "unhappy", "hate", "awful", "poor", "disappointing", "frustrating", "horrible", "stressful"}

        found_positives = [kw for kw in positive_keywords if kw in text_lower]
        found_negatives = [kw for kw in negative_keywords if kw in text_lower]

        if found_positives and not found_negatives:
            sentiment = "positive"
            # Simple scoring: more positive keywords, higher confidence
            score = 0.6 + (0.05 * len(found_positives))
            score = min(score, 0.95) # Cap the score
            analysis_details.append(f"Detected positive indicators: {', '.join(found_positives)}.")
        elif found_negatives and not found_positives:
            sentiment = "negative"
            # Simple scoring: more negative keywords, higher confidence
            score = 0.6 + (0.05 * len(found_negatives))
            score = min(score, 0.95) # Cap the score
            analysis_details.append(f"Detected negative indicators: {', '.join(found_negatives)}.")
        elif found_positives and found_negatives:
            sentiment = "mixed"
            # In case of mixed sentiment, lean towards neutral or slightly biased
            if len(found_positives) > len(found_negatives):
                score = 0.55
            elif len(found_negatives) > len(found_positives):
                score = 0.45
            else:
                score = 0.5
            analysis_details.append(
                f"Detected mixed indicators - Positive: {', '.join(found_positives)}; "
                f"Negative: {', '.join(found_negatives)}."
            )
        else:
            sentiment = "neutral"
            score = 0.5
            analysis_details.append("No strong sentiment indicators detected.")

        result = {
            "sentiment": sentiment,
            "score": round(score, 2),
            "node_name": self.node_name,
            "details": " ".join(analysis_details)
        }

        logger.info(f"[{self.node_name}] Analysis complete. Sentiment: '{sentiment}', Score: {score:.2f}.")
        logger.debug(f"[{self.node_name}] Full analysis result: {result}")

        return result