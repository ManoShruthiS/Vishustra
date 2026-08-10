import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists and BaseNode is there
# This import path is specified by the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra node that simulates sentiment analysis on input text.

    This node takes a string as input data and returns a dictionary
    containing the identified sentiment (positive, negative, or neutral)
    and a simulated sentiment score.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this sentiment analyzer node.
        """
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        The 'data' is expected to be a string. The method simulates
        sentiment analysis based on a simple keyword matching approach
        and returns a structured dictionary with the results.

        Args:
            data: The input text (expected to be a string) to analyze for sentiment.
            context: A dictionary containing contextual information for the node's
                     operation. Not heavily used in this simulated version, but
                     available for future extensions (e.g., model configurations,
                     language settings).

        Returns:
            A dictionary containing the original text, the determined sentiment
            ("positive", "negative", "neutral"), and a simulated sentiment score.
            Example: {"text": "This is great!", "sentiment": "positive", "score": 0.8}

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the input 'data' is an empty string after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Starting sentiment analysis for data type: {type(data).__name__}.")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_to_analyze = data.strip()
        if not text_to_analyze:
            error_msg = f"[{self.node_name}] Input text is empty or contains only whitespace after stripping."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Simulate sentiment analysis using a simple keyword-based approach.
        # In a real-world Vishustra application, this would integrate with an actual
        # NLP library (e.g., NLTK, spaCy, Hugging Face transformers) or an external
        # sentiment analysis service (e.g., Azure Cognitive Services, AWS Comprehend).
        text_lower = text_to_analyze.lower()

        positive_indicators = {"great", "good", "excellent", "love", "happy", "awesome", "fantastic", "wonderful", "amazing"}
        negative_indicators = {"bad", "terrible", "horrible", "hate", "sad", "awful", "disappointing", "poor", "frustrating"}

        found_positive = any(word in text_lower for word in positive_indicators)
        found_negative = any(word in text_lower for word in negative_indicators)

        sentiment = "neutral"
        score = 0.0 # Default score for neutral sentiment

        if found_positive and not found_negative:
            sentiment = "positive"
            score = 0.8 # Simulated fixed score for positive sentiment
        elif found_negative and not found_positive:
            sentiment = "negative"
            score = -0.7 # Simulated fixed score for negative sentiment
        elif found_positive and found_negative:
            # When both positive and negative indicators are present,
            # for this simple simulation, we'll assign a neutral sentiment.
            # A more sophisticated node might implement a weighted score or
            # use a more advanced model to resolve conflicting signals.
            sentiment = "neutral"
            score = 0.0
            logger.warning(f"[{self.node_name}] Mixed sentiment indicators found for text starting with '{text_to_analyze[:70]}...'. Defaulting to neutral.")
        else:
            logger.debug(f"[{self.node_name}] No strong positive or negative indicators found for text starting with '{text_to_analyze[:70]}...'. Assigning neutral.")
        
        result = {
            "text": data, # Return the original, unstripped data as part of the result
            "sentiment": sentiment,
            "score": score
        }

        logger.info(f"[{self.node_name}] Processed text (first 70 chars): '{text_to_analyze[:70]}...', Result: Sentiment='{sentiment}', Score={score:.2f}")
        return result