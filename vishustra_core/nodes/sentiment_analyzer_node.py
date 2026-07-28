import logging
from typing import Any, Dict, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node designed to perform sentiment analysis on input text.

    This node takes a string as input, analyzes its content for
    positive and negative indicators, and outputs a classified sentiment
    along with a simulated score. The analysis is performed using a
    configurable keyword-matching approach, suitable for demonstration
    and basic sentiment detection within a pipeline.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the SentimentAnalyzerNode with optional configuration.

        Args:
            config: An optional dictionary for node-specific configuration.
                    Expected keys:
                    - 'positive_keywords': A list of strings indicating positive sentiment.
                                           Defaults to a standard set if not provided.
                    - 'negative_keywords': A list of strings indicating negative sentiment.
                                           Defaults to a standard set if not provided.
        """
        self._node_name = "SentimentAnalyzerNode"
        self._config = config if config is not None else {}

        # Default keyword sets for simulation
        self._positive_keywords: List[str] = self._config.get(
            'positive_keywords',
            ["good", "great", "excellent", "happy", "love", "joy", "fantastic", "amazing", "wonderful"]
        )
        self._negative_keywords: List[str] = self._config.get(
            'negative_keywords',
            ["bad", "terrible", "horrible", "sad", "hate", "anger", "awful", "frustrating", "poor"]
        )
        logger.debug(f"[{self.node_name}] Initialized with custom configuration: {self._config}")
        logger.debug(f"[{self.node_name}] Positive keywords: {self._positive_keywords}")
        logger.debug(f"[{self.node_name}] Negative keywords: {self._negative_keywords}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return self._node_name

    def _analyze_sentiment_keyword(self, text: str) -> Tuple[str, float]:
        """
        Simulates sentiment analysis using a simple keyword-matching algorithm.
        Counts occurrences of positive and negative keywords to derive a sentiment.

        Args:
            text: The input string to analyze.

        Returns:
            A tuple containing:
            - sentiment_label (str): The classified sentiment ("positive", "negative", "neutral").
            - score (float): A numerical score (positive for positive sentiment, negative for negative).
        """
        lower_text = text.lower()
        words = lower_text.split()

        positive_count = sum(1 for word in words if word in self._positive_keywords)
        negative_count = sum(1 for word in words if word in self._negative_keywords)

        score = float(positive_count - negative_count)

        if score > 0:
            sentiment_label = "positive"
        elif score < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        logger.debug(
            f"[{self.node_name}] Text snippet '{text[:75]}...' analyzed. "
            f"Positive matches: {positive_count}, Negative matches: {negative_count}, "
            f"Raw score: {score}, Label: {sentiment_label}"
        )
        return sentiment_label, score

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The `data` input is expected to be a string containing the text
        to be analyzed. The `context` dictionary can provide additional
        runtime information, though this simulation primarily uses `data`.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary providing additional context for processing.
                     (e.g., 'language' for future enhancements, though not used here).

        Returns:
            A dictionary containing the sentiment analysis results:
            - 'original_text': The input text that was analyzed.
            - 'sentiment': The classified sentiment ("positive", "negative", or "neutral").
            - 'score': A numerical score representing the sentiment. Positive values
                       indicate positive sentiment, negative values indicate negative.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        logger.info(f"[{self.node_name}] Initiating sentiment analysis for incoming data.")
        
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type encountered. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for sentiment analysis. "
                f"Received type: {type(data).__name__}."
            )

        stripped_data = data.strip()
        if not stripped_data:
            logger.error(f"[{self.node_name}] Input data is an empty or whitespace-only string after stripping.")
            raise ValueError(
                f"[{self.node_name}] Input 'data' cannot be an empty string or consist only of whitespace "
                f"for sentiment analysis."
            )

        # Log context information, a real implementation might use context['language']
        # to select a language-specific model or modify keyword sets dynamically.
        if context:
            logger.debug(f"[{self.node_name}] Context received: {context}")

        sentiment_label, score = self._analyze_sentiment_keyword(stripped_data)

        result = {
            "original_text": data, # Return original data, not stripped
            "sentiment": sentiment_label,
            "score": score
        }
        logger.info(f"[{self.node_name}] Sentiment analysis successfully completed. Result: {result}")
        return result