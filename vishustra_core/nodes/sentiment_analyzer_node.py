import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to analyze the sentiment of input text data.

    This node expects a string as input, performs a simulated sentiment analysis
    based on a simple keyword detection mechanism, and returns a structured
    dictionary containing the original text, its determined sentiment
    ('positive', 'negative', 'neutral'), and a simulated confidence score.

    In a production environment, this node would integrate with an actual
    sentiment analysis model or service.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the provided text data.

        Args:
            data (Any): The input data to be analyzed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. Not directly used by
                                       this node's current implementation but available.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis result:
                            - 'text': The original input string.
                            - 'sentiment': The classified sentiment ('positive', 'negative', 'neutral').
                            - 'score': A simulated confidence score for the sentiment, ranging from 0.0 to 1.0.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input string is empty or consists only of whitespace.
        """
        if not isinstance(data, str):
            error_message = f"Invalid input type for SentimentAnalyzerNode. Expected string, got {type(data).__name__}."
            logger.error(error_message)
            raise TypeError(error_message)
        
        trimmed_text = data.strip()
        if not trimmed_text:
            error_message = "Input string for SentimentAnalyzerNode cannot be empty or solely whitespace."
            logger.warning(error_message)
            raise ValueError(error_message)

        text_lower = trimmed_text.lower()
        
        # Simple keyword-based sentiment simulation
        positive_keywords = ["good", "great", "excellent", "amazing", "happy", "love", "awesome", "fantastic"]
        negative_keywords = ["bad", "terrible", "poor", "awful", "hate", "disappointing", "frustrating", "horrible"]

        is_positive = any(keyword in text_lower for keyword in positive_keywords)
        is_negative = any(keyword in text_lower for keyword in negative_keywords)

        sentiment = "neutral"
        score = 0.50  # Default neutral score

        if is_negative and not is_positive:
            sentiment = "negative"
            score = 0.90 if any(kw in text_lower for kw in negative_keywords[:4]) else 0.75
        elif is_positive and not is_negative:
            sentiment = "positive"
            score = 0.90 if any(kw in text_lower for kw in positive_keywords[:4]) else 0.75
        elif is_positive and is_negative:
            # When both positive and negative keywords are present, lean towards neutral or slightly negative
            sentiment = "neutral" 
            score = 0.55 # Acknowledging mixed signals
            logger.info(f"Mixed sentiment indicators detected for text (first 50 chars): '{trimmed_text[:50]}...'. Classified as neutral.")
        else:
            logger.debug(f"No strong sentiment keywords found for text (first 50 chars): '{trimmed_text[:50]}...'. Classified as neutral.")
            # Keep default neutral sentiment and score

        result = {
            "text": data,  # Retain the original (untrimmed) input text
            "sentiment": sentiment,
            "score": round(score, 2)
        }
        
        logger.info(f"Processed text for sentiment (first 50 chars): '{trimmed_text[:50]}...' -> Sentiment: {sentiment}, Score: {score:.2f}")
        
        return result