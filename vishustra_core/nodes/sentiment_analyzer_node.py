import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists at this path
# In a real project, this would likely be an absolute import
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to analyze the sentiment of input text data.

    This node simulates sentiment analysis by identifying positive and negative
    keywords within the input text and computing a sentiment score and label.
    In a production environment, this would integrate with a robust
    natural language processing (NLP) library or a dedicated sentiment model.
    """

    def __init__(self):
        """
        Initializes the SentimentAnalyzerNode.
        Sets up predefined keyword lists for simulated sentiment detection.
        """
        # In a real scenario, a pre-trained sentiment model or an external
        # NLP service client would be initialized here.
        self._positive_keywords = {
            "good", "great", "excellent", "love", "happy", "awesome",
            "fantastic", "amazing", "wonderful", "brilliant", "joy", "success"
        }
        self._negative_keywords = {
            "bad", "terrible", "awful", "hate", "unhappy", "poor",
            "frustrating", "disappointing", "horrible", "ugly", "failure", "stress"
        }
        logger.debug(f"[{self.node_name}] Initialized with {len(self._positive_keywords)} positive and {len(self._negative_keywords)} negative keywords.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data.

        This method expects a string as input. It counts occurrences of
        predefined positive and negative keywords to determine a sentiment
        (positive, negative, or neutral) and a normalized score.

        Args:
            data (Any): The input data, expected to be a string (text) for analysis.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing flow (e.g., flow_id).

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "original_text": The input text.
                            - "sentiment": The detected sentiment ('positive', 'negative', 'neutral').
                            - "score": A numeric score ranging from -1.0 (most negative) to 1.0 (most positive).

        Raises:
            TypeError: If the input `data` is not a string.
        """
        flow_id = context.get('flow_id', 'N/A')
        logger.info(f"[{self.node_name}] Starting sentiment analysis for flow_id: {flow_id}.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type for flow_id '{flow_id}'. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_lower = data.lower()
        
        positive_count = sum(1 for keyword in self._positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self._negative_keywords if keyword in text_lower)

        sentiment = "neutral"
        score = 0.0

        total_keyword_mentions = positive_count + negative_count

        if total_keyword_mentions > 0:
            # Calculate a normalized score between -1.0 and 1.0
            score = (positive_count - negative_count) / total_keyword_mentions
            
            # Determine sentiment label based on score thresholds
            if score > 0.1:  # Threshold for positive sentiment
                sentiment = "positive"
            elif score < -0.1:  # Threshold for negative sentiment
                sentiment = "negative"
            else:
                sentiment = "neutral"  # Score close to zero
        else:
            # If no sentiment keywords are found, default to neutral sentiment.
            logger.debug(
                f"[{self.node_name}] No relevant keywords found in text for flow_id '{flow_id}'. "
                "Defaulting to neutral sentiment."
            )

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "score": round(score, 4)  # Round score for cleaner output
        }

        logger.debug(
            f"[{self.node_name}] Analysis complete for flow_id '{flow_id}'. "
            f"Text snippet: '{data[:75]}{'...' if len(data) > 75 else ''}'. "
            f"Sentiment: '{sentiment}', Score: {result['score']}"
        )
        return result