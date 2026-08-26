import logging
from typing import Any, Dict

# Assuming BaseNode is correctly importable from this path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that simulates sentiment analysis on input text.

    This node takes a string as input and returns a dictionary containing
    the original text, a simulated sentiment label (positive, negative, neutral),
    and a corresponding sentiment score.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Expected `data` type: str
        Expected `context` type: Dict[str, Any] (not used in this simulation)

        Returns:
            Dict[str, Any]: A dictionary with 'original_text', 'sentiment', and 'score'.
                            Example: {'original_text': 'I love this!', 'sentiment': 'positive', 'score': 0.9}

        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"SentimentAnalyzerNode received non-string data. Expected str, got {type(data).__name__}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = data.lower()
        sentiment = "neutral"
        score = 0.0

        # Simulate sentiment analysis based on keywords
        if any(keyword in text for keyword in ["great", "awesome", "love", "excellent", "happy", "positive", "wonderful"]):
            sentiment = "positive"
            score = 0.85
        elif any(keyword in text for keyword in ["bad", "terrible", "hate", "awful", "unhappy", "negative", "horrible"]):
            sentiment = "negative"
            score = -0.75
        else:
            sentiment = "neutral"
            score = 0.0

        if score != 0.0: # Add some variance for non-zero scores
            score += (text.count('!') * 0.05) if sentiment == "positive" else -(text.count('!') * 0.05)
            score = max(-1.0, min(1.0, score)) # Clamp score between -1 and 1

        result = {
            "original_text": data,
            "sentiment": sentiment,
            "score": score
        }

        logger.info(f"Analyzed sentiment for '{data[:50]}...' -> {sentiment} (score: {score:.2f})")
        return result

# Optional: Example usage (for testing purposes, not part of Vishustra execution flow)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    analyzer = SentimentAnalyzerNode()
    dummy_context = {}

    try:
        # Test cases
        print(f"Node Name: {analyzer.node_name}")
        
        text1 = "This is a great product, I love it!"
        result1 = analyzer.process(text1, dummy_context)
        print(f"Text: '{text1}' -> {result1}")

        text2 = "I am quite disappointed, this is terrible."
        result2 = analyzer.process(text2, dummy_context)
        print(f"Text: '{text2}' -> {result2}")

        text3 = "The weather today is neither good nor bad."
        result3 = analyzer.process(text3, dummy_context)
        print(f"Text: '{text3}' -> {result3}")

        text4 = "An amazing experience!!!!"
        result4 = analyzer.process(text4, dummy_context)
        print(f"Text: '{text4}' -> {result4}")
        
        # Test error handling
        try:
            analyzer.process(123, dummy_context)
        except ValueError as e:
            print(f"Caught expected error: {e}")

        try:
            analyzer.process(None, dummy_context)
        except ValueError as e:
            print(f"Caught expected error: {e}")

    except Exception as e:
        logger.exception("An unexpected error occurred during example usage.")
        print(f"An unexpected error occurred: {e}")
