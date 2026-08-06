import logging
from typing import Any, Dict, Union

# Assuming BaseNode is accessible via this absolute import path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to perform sentiment analysis on input text data.

    This implementation uses a heuristic keyword-based approach for demonstration purposes.
    In a production environment, this would integrate with a robust NLP library or a
    pre-trained machine learning model to provide more accurate sentiment scores.
    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzerNode with predefined keyword lists for
        simulated sentiment detection. These keywords are case-insensitive
        during processing.
        """
        self._positive_keywords = {
            "good", "great", "excellent", "awesome", "happy", "love", "positive",
            "superb", "fantastic", "amazing", "wonderful", "brilliant", "joy"
        }
        self._negative_keywords = {
            "bad", "terrible", "horrible", "awful", "unhappy", "hate", "negative",
            "poor", "frustrating", "disappointing", "sad", "angry"
        }
        # Neutral keywords are not explicitly used for scoring but can be part of
        # text processing or stop word removal in more advanced scenarios.

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "SentimentAnalyzerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Analyzes the sentiment of the input `data`.

        Expects `data` to be a string of text. It tokenizes the text and
        counts occurrences of predefined positive and negative keywords
        to assign a sentiment label and a score.

        Args:
            data: The input text data to be analyzed. Must be a string.
            context: A dictionary containing contextual information relevant
                     to the current processing pipeline. (Not directly used
                     in this specific node's logic but part of the standard API).

        Returns:
            A dictionary containing the sentiment analysis result.
            Example successful output: `{"sentiment": "positive", "score": 0.85}`.
            Example error output: `{"error": "Invalid input data type..."}`.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            return {"error": error_msg}

        text = data.lower()
        if not text.strip():
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only text for "
                f"sentiment analysis. Returning neutral sentiment."
            )
            return {"sentiment": "neutral", "score": 0.0}

        positive_count = 0
        negative_count = 0
        
        # Simple word tokenization. For real-world, use NLP tokenizers.
        words = text.split() 

        for word in words:
            if word in self._positive_keywords:
                positive_count += 1
            elif word in self._negative_keywords:
                negative_count += 1

        sentiment: str
        score: float

        total_meaningful_words = positive_count + negative_count

        if total_meaningful_words == 0:
            sentiment = "neutral"
            score = 0.0
            logger.info(f"[{self.node_name}] No sentiment-bearing keywords found. Result: {sentiment}, Score: {score:.2f}")
        elif positive_count > negative_count:
            sentiment = "positive"
            # Score reflects the intensity relative to overall sentiment-bearing words
            score = round((positive_count - negative_count) / total_meaningful_words, 2)
            logger.info(f"[{self.node_name}] Detected positive sentiment. Result: {sentiment}, Score: {score:.2f}")
        elif negative_count > positive_count:
            sentiment = "negative"
            score = round((negative_count - positive_count) / total_meaningful_words, 2)
            logger.info(f"[{self.node_name}] Detected negative sentiment. Result: {sentiment}, Score: {score:.2f}")
        else:  # positive_count == negative_count and total_meaningful_words > 0
            sentiment = "neutral"
            score = 0.0 # Equal positive and negative cancels out
            logger.info(f"[{self.node_name}] Balanced sentiment keywords. Result: {sentiment}, Score: {score:.2f}")

        return {"sentiment": sentiment, "score": score}