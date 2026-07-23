import logging
import random
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that analyzes the sentiment of a given text.
    It simulates sentiment analysis without relying on external NLP libraries.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its sentiment.

        Expected `data` input: A string containing the text to be analyzed.
        Expected `context` input: A dictionary (not directly used by this node's logic,
                                  but available for potential future extensions).

        Args:
            data (Any): The input data, expected to be a string of text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - 'text': The original input text.
                            - 'sentiment': A string indicating the sentiment ("positive",
                                           "negative", or "neutral").
                            - 'score': A float representing the sentiment strength,
                                       ranging from -1.0 (strongly negative) to 1.0
                                       (strongly positive).

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', got '%s'. Data: %s",
                self.node_name,
                type(data).__name__,
                data
            )
            raise TypeError(f"SentimentAnalyzerNode expects string input, got {type(data).__name__}")

        text = data.lower()
        sentiment = "neutral"
        score = random.uniform(-0.2, 0.2)  # Default neutral score

        positive_keywords = {"good", "great", "excellent", "wonderful", "love", "happy", "positive", "amazing", "fantastic", "brilliant"}
        negative_keywords = {"bad", "terrible", "horrible", "awful", "hate", "negative", "poor", "unhappy", "frustrating", "disappointing"}

        pos_count = sum(1 for word in text.split() if word in positive_keywords)
        neg_count = sum(1 for word in text.split() if word in negative_keywords)

        if pos_count > neg_count:
            sentiment = "positive"
            score = random.uniform(0.6, 1.0)  # Strong positive
        elif neg_count > pos_count:
            sentiment = "negative"
            score = random.uniform(-1.0, -0.6)  # Strong negative
        elif pos_count == 0 and neg_count == 0:
            sentiment = "neutral"
            score = random.uniform(-0.2, 0.2) # Genuinely neutral if no keywords
        else: # Equal counts of positive and negative keywords, or non-zero and equal
            sentiment = "neutral"
            score = random.uniform(-0.4, 0.4) # Slightly less sure neutral

        result = {
            "text": data,
            "sentiment": sentiment,
            "score": round(score, 4)
        }

        logger.info(
            "[%s] Processed text sentiment: '%s' (score: %.4f) for input text (truncated): '%s'",
            self.node_name,
            result["sentiment"],
            result["score"],
            data[:50] + "..." if len(data) > 50 else data
        )
        return result

# Example of how to use this node (for testing/demonstration purposes, not part of the class itself)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sentiment_node = SentimentAnalyzerNode()

    test_texts = [
        "This is an absolutely wonderful product! I love it.",
        "The service was terrible and I am very unhappy with the experience.",
        "It's an average day, nothing special happened.",
        "The documentation is good but the performance is bad.",
        "",
        "Fantastic brilliant excellent, very good!",
        "Horrible, awful, terrible, bad.",
        123  # This should trigger an error
    ]

    for i, text_input in enumerate(test_texts):
        print(f"\n--- Test Case {i+1} ---")
        try:
            result = sentiment_node.process(text_input, {})
            print(f"Input: '{result['text']}'")
            print(f"Sentiment: {result['sentiment']} (Score: {result['score']})")
        except TypeError as e:
            print(f"Error processing input '{text_input}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")