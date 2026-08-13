from typing import Any, Dict
import logging

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that performs sentiment analysis on input text.
    It simulates sentiment detection by looking for predefined positive and negative keywords.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.

        Expected `data` type: str
        Returns: A dictionary containing 'sentiment' (str) and 'score' (float).
                 Sentiment can be 'positive', 'negative', or 'neutral'.
                 Score ranges from -1.0 (highly negative) to 1.0 (highly positive).
        Raises: ValueError if the input `data` is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"SentimentAnalyzerNode received invalid data type: {type(data)}. Expected string."
            logger.error(error_msg)
            raise ValueError(error_msg)

        text = data.lower()
        sentiment_score = 0.0
        sentiment_label = "neutral"

        # Simple keyword-based sentiment simulation
        positive_keywords = ["good", "great", "excellent", "awesome", "happy", "love", "positive", "fine", "well"]
        negative_keywords = ["bad", "terrible", "poor", "awful", "unhappy", "hate", "negative", "problem", "wrong"]

        found_positive = sum(text.count(kw) for kw in positive_keywords)
        found_negative = sum(text.count(kw) for kw in negative_keywords)

        if found_positive > found_negative:
            sentiment_label = "positive"
            sentiment_score = min(1.0, 0.2 * (found_positive - found_negative)) # Simple score scaling
        elif found_negative > found_positive:
            sentiment_label = "negative"
            sentiment_score = max(-1.0, -0.2 * (found_negative - found_positive)) # Simple score scaling
        else:
            sentiment_label = "neutral"
            sentiment_score = 0.0

        result = {
            "sentiment": sentiment_label,
            "score": sentiment_score
        }

        logger.debug(f"[{self.node_name}] Processed text (len={len(data)}). Result: {result}")
        return result

# Example Usage (for testing purposes, not part of the node itself)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    analyzer = SentimentAnalyzerNode()

    test_cases = [
        "This is a great product, I love it!",
        "The service was terrible and very poor.",
        "It's an average experience, nothing special.",
        "I am happy with the outcome.",
        "This is wrong and bad.",
        "",
        "Good good good bad", # mixed
    ]

    for text_input in test_cases:
        try:
            analysis_result = analyzer.process(text_input, {})
            print(f"Text: '{text_input}'\n  -> {analysis_result}\n")
        except ValueError as e:
            print(f"Error processing '{text_input}': {e}\n")

    # Test with invalid input type
    try:
        analyzer.process(123, {})
    except ValueError as e:
        print(f"Error processing non-string data: {e}\n")

    try:
        analyzer.process(["list", "of", "strings"], {})
    except ValueError as e:
        print(f"Error processing list data: {e}\n")