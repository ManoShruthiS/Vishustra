import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for the module
logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to evaluate 
    the emotional tone of textual data. 
    
    This node processes raw strings and returns a structured dictionary 
    containing the sentiment classification and a confidence score.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the input data to determine sentiment.

        Args:
            data: The input payload, expected to be a string representing text.
            context: Shared execution context containing pipeline metadata.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'original_text': The input processed.
                - 'sentiment': 'positive', 'negative', or 'neutral'.
                - 'confidence': A float representing the heuristic confidence level.

        Raises:
            TypeError: If the input data is not a string.
            Exception: For unexpected processing errors.
        """
        logger.info(f"Node [{self.node_name}] started processing.")

        if not isinstance(data, str):
            error_msg = f"SentimentAnalyzer expected string input, received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Clean input data
            clean_text = data.strip().lower()
            
            # Simple heuristic-based simulation of sentiment analysis
            # In production, this would interface with a pre-trained model or LLM service
            positive_keywords = {"excellent", "great", "good", "happy", "efficient", "robust"}
            negative_keywords = {"bad", "poor", "error", "slow", "terrible", "failure"}

            pos_count = sum(1 for word in positive_keywords if word in clean_text)
            neg_count = sum(1 for word in negative_keywords if word in clean_text)

            if pos_count > neg_count:
                sentiment = "positive"
                confidence = min(0.5 + (pos_count * 0.1), 0.99)
            elif neg_count > pos_count:
                sentiment = "negative"
                confidence = min(0.5 + (neg_count * 0.1), 0.99)
            else:
                sentiment = "neutral"
                confidence = 0.5

            result = {
                "original_text": data,
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "metadata": {
                    "processor": self.node_name,
                    "version": "1.0.0"
                }
            }

            logger.info(f"Node [{self.node_name}] successfully categorized text as '{sentiment}'.")
            return result

        except Exception as e:
            logger.exception(f"Unexpected error in {self.node_name} during processing.")
            raise e