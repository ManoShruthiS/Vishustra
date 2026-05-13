import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A node designed to evaluate the emotional tone of a given text input.
    Provides a sentiment score and a categorical label (Positive, Negative, Neutral).
    """

    def __init__(self, threshold: float = 0.05):
        self._threshold = threshold
        # Simulated lexicon for the sake of the demonstration
        self._positive_words = {"excellent", "good", "great", "efficient", "robust", "love", "amazing"}
        self._negative_words = {"bad", "poor", "slow", "error", "fail", "bug", "terrible"}

    @property
    def node_name(self) -> str:
        """Returns the identifier for this specific node type."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.
        
        Args:
            data: Expected to be a string or a dictionary containing a 'text' key.
            context: Execution context containing metadata or shared state.

        Returns:
            A dictionary containing the sentiment score and label.
        
        Raises:
            ValueError: If the input data format is unsupported or empty.
        """
        logger.info(f"Node '{self.node_name}' starting execution.")

        try:
            text = self._extract_text(data)
            if not text:
                raise ValueError("Input text is empty or null.")

            sentiment_score = self._calculate_score(text)
            sentiment_label = self._get_label(sentiment_score)

            result = {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "processed_length": len(text)
            }

            logger.debug(f"Sentiment analysis complete: {sentiment_label} ({sentiment_score})")
            return result

        except Exception as e:
            logger.error(f"Error in SentimentAnalyzerNode: {str(e)}", exc_info=True)
            raise

    def _extract_text(self, data: Any) -> str:
        """Helper to normalize input data into a processable string."""
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict) and "text" in data:
            return str(data["text"]).strip()
        
        raise TypeError(f"SentimentAnalyzer expected str or dict containing 'text', got {type(data)}")

    def _calculate_score(self, text: str) -> float:
        """
        Simulates a sentiment calculation. 
        Returns a float between -1.0 (Negative) and 1.0 (Positive).
        """
        words = text.lower().split()
        if not words:
            return 0.0

        pos_count = sum(1 for word in words if word in self._positive_words)
        neg_count = sum(1 for word in words if word in self._negative_words)
        
        total_relevant = pos_count + neg_count
        if total_relevant == 0:
            return 0.0
            
        return (pos_count - neg_count) / total_relevant

    def _get_label(self, score: float) -> str:
        """Maps a numerical score to a categorical label."""
        if score > self._threshold:
            return "POSITIVE"
        elif score < -self._threshold:
            return "NEGATIVE"
        return "NEUTRAL"