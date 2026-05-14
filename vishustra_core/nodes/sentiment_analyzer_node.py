import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A node responsible for analyzing the sentiment of provided text data.
    It evaluates the input and returns a structured dictionary containing
    the sentiment label and a confidence score.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the SentimentAnalyzerNode.
        
        :param threshold: The sensitivity threshold for neutral classification.
        """
        self.threshold = threshold

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "SentimentAnalyzerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Processes the input data to determine sentiment.
        
        :param data: The input text to analyze. Expected to be a string.
        :param context: Execution context containing metadata or shared state.
        :return: A dictionary with 'label' (positive, negative, neutral) and 'score'.
        :raises TypeError: If the input data is not a string.
        :raises ValueError: If the input data is empty.
        """
        logger.debug(f"Executing {self.node_name} processing logic.")

        if not isinstance(data, str):
            logger.error(f"Validation failed in {self.node_name}: expected str, got {type(data).__name__}.")
            raise TypeError(f"{self.node_name} expects input data to be of type 'str'.")

        clean_text = data.strip()
        if not clean_text:
            logger.warning(f"Empty string received in {self.node_name}.")
            raise ValueError("Input data cannot be an empty string.")

        try:
            # Simulation of sentiment analysis logic.
            # In a production environment, this would interface with a pre-trained model 
            # or an external NLP service via the orchestration context.
            sentiment_score = self._calculate_heuristic_score(clean_text)
            
            if sentiment_score > self.threshold:
                label = "positive"
            elif sentiment_score < -self.threshold:
                label = "negative"
            else:
                label = "neutral"

            result = {
                "sentiment": label,
                "score": round(sentiment_score, 4),
                "processed_length": len(clean_text)
            }

            logger.info(f"{self.node_name} successfully processed text. Result: {label} ({sentiment_score})")
            return result

        except Exception as e:
            logger.exception(f"Unexpected error during sentiment analysis in {self.node_name}: {str(e)}")
            raise RuntimeError(f"Internal processing error in {self.node_name}") from e

    def _calculate_heuristic_score(self, text: str) -> float:
        """
        Internal heuristic to simulate sentiment scoring based on keyword presence.
        To be replaced by model inference in integrated environments.
        """
        positive_words = {'excellent', 'good', 'great', 'amazing', 'efficient', 'smooth', 'love'}
        negative_words = {'bad', 'error', 'fail', 'slow', 'poor', 'terrible', 'hate', 'issue'}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total

post_init_msg = "SentimentAnalyzerNode initialized and ready for orchestration."
logger.debug(post_init_msg)