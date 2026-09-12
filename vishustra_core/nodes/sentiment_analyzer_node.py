import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node designed to perform sentiment analysis on input text.

    This node simulates sentiment analysis by identifying simple keywords
    within the text to determine a sentiment label (positive, negative, neutral)
    and an associated score. It supports processing either a single string
    or a list of strings, providing a structured result for each.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzerNode"

    def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        Internal method to perform sentiment analysis for a single piece of text.
        This implementation uses a simplified keyword-based approach for demonstration.

        Args:
            text: The text string to analyze.

        Returns:
            A dictionary containing the original text, its determined sentiment
            ('positive', 'negative', or 'neutral'), and a numerical score.
        """
        text_lower = text.lower()
        score = 0
        sentiment = "neutral"

        # Simple keyword-based sentiment detection
        positive_keywords = ["good", "great", "excellent", "happy", "love", "awesome", "fantastic", "wonderful"]
        negative_keywords = ["bad", "terrible", "poor", "unhappy", "hate", "awful", "horrible", "frustrating"]

        for keyword in positive_keywords:
            if keyword in text_lower:
                score += 1
        for keyword in negative_keywords:
            if keyword in text_lower:
                score -= 1

        if score > 0:
            sentiment = "positive"
        elif score < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "text": text,
            "sentiment": sentiment,
            "score": score
        }

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Processes the input data to determine sentiment.

        The `data` input is expected to be either a single string or a list of strings.
        Each string will be analyzed for sentiment, and a structured result will be returned.

        Args:
            data: The input data, which can be a single string or a list of strings.
                  Empty strings or lists containing non-string elements will trigger
                  error handling.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context but adheres
                     to the `BaseNode` interface.

        Returns:
            If `data` is a single string, a dictionary containing sentiment analysis
            results for that string. If `data` is a list of strings, a list of
            dictionaries, where each dictionary corresponds to the analysis of
            an input string.

            Each result dictionary will include:
            - 'text': The original input text.
            - 'sentiment': The determined sentiment ('positive', 'negative', 'neutral').
            - 'score': A numerical representation of the sentiment.
            - Optionally, an 'error' key if an issue was encountered for a specific item.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings,
                       or if a list contains non-string elements.
            ValueError: If a non-empty string input is provided but evaluates to empty
                        after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Initiating sentiment analysis for input of type: {type(data)}")

        if isinstance(data, str):
            processed_text = data.strip()
            if not processed_text:
                logger.warning(f"[{self.node_name}] Received an empty or whitespace-only string for analysis.")
                raise ValueError("Input text cannot be empty or consist only of whitespace.")
            return self._analyze_single_text(processed_text)

        elif isinstance(data, list):
            results = []
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        f"[{self.node_name}] Input list contains a non-string element at index {i}: "
                        f"Expected str, got {type(item).__name__}."
                    )
                    raise TypeError(
                        f"All elements in the input list for {self.node_name} must be strings. "
                        f"Found {type(item).__name__} at index {i}."
                    )

                processed_item = item.strip()
                if not processed_item:
                    logger.warning(
                        f"[{self.node_name}] Skipping empty or whitespace-only string at index {i} "
                        "during list processing."
                    )
                    results.append({"text": item, "sentiment": "neutral", "score": 0, "error": "Empty string input"})
                    continue
                results.append(self._analyze_single_text(processed_item))
            return results

        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type: {type(data).__name__}. "
                "Expected str or List[str]."
            )
            raise TypeError(
                f"Invalid input data type for {self.node_name}. "
                f"Expected str or List[str], got {type(data).__name__}."
            )