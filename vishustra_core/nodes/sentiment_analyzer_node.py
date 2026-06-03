import logging
from typing import Any, Dict

# Assuming vishustra_core is a package and nodes is a subpackage
# where base_node is located.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A Vishustra processing node that performs a simulated sentiment analysis
    on textual data.

    This node expects input `data` to be either a string or a dictionary.
    If `data` is a dictionary, it looks for a text field specified by
    'sentiment_input_key' in the `context` (defaults to 'text')
    to extract the text for analysis.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input text data using a keyword-based
        simulation.

        The method handles various input data types and provides robust error
        handling and logging.

        Args:
            data: The input data, expected to be a string directly, or a dictionary
                  containing a string under a specific key.
            context: A dictionary containing runtime configuration.
                     - 'sentiment_input_key' (str, optional): Specifies the key
                       in the input `data` dictionary that holds the text to be
                       analyzed. Defaults to 'text'.

        Returns:
            A dictionary containing the sentiment analysis results, structured as:
            ```json
            {
                'original_input': data,
                'sentiment_analysis': {
                    'label': 'positive' | 'negative' | 'neutral',
                    'score': float,
                    'details': str
                }
            }
            ```
            If an error prevents analysis, the dictionary will contain an 'error' key.
        """
        text_to_analyze: str = ""
        sentiment_input_key = context.get('sentiment_input_key', 'text')

        # --- Input Data Validation and Extraction ---
        if isinstance(data, str):
            text_to_analyze = data
            logger.debug(f"[{self.node_name}] Processing string input directly.")
        elif isinstance(data, dict):
            # Attempt to extract text from the dictionary using the configured key
            text_value = data.get(sentiment_input_key)
            if text_value is None:
                logger.warning(
                    f"[{self.node_name}] Input dictionary is missing the expected key "
                    f"'{sentiment_input_key}' for sentiment analysis. "
                    f"Available keys: {list(data.keys()) if data else 'None'}"
                )
                return {
                    'original_input': data,
                    'error': f"Missing text field '{sentiment_input_key}' in input dictionary."
                }
            if not isinstance(text_value, str):
                logger.error(
                    f"[{self.node_name}] Value for key '{sentiment_input_key}' is not a string. "
                    f"Type found: {type(text_value).__name__}."
                )
                return {
                    'original_input': data,
                    'error': f"Expected text for '{sentiment_input_key}' to be a string, "
                             f"but found {type(text_value).__name__}."
                }
            text_to_analyze = text_value
            logger.debug(
                f"[{self.node_name}] Extracted text from dictionary using key "
                f"'{sentiment_input_key}'."
            )
        else:
            # Handle unsupported data types
            logger.error(
                f"[{self.node_name}] Invalid input data type for sentiment analysis. "
                f"Expected str or dict, got {type(data).__name__}."
            )
            return {
                'original_input': data,
                'error': f"Unsupported data type: {type(data).__name__}. "
                         f"Expected string or dictionary."
            }

        # --- Handle empty or whitespace-only text ---
        if not text_to_analyze.strip():
            logger.info(
                f"[{self.node_name}] Received empty or whitespace-only text for analysis. "
                f"Returning neutral sentiment."
            )
            return {
                'original_input': data,
                'sentiment_analysis': {
                    'label': 'neutral',
                    'score': 0.0,
                    'details': 'Empty or whitespace-only text provided.'
                }
            }

        # --- Simulated Sentiment Analysis Logic ---
        # This is a basic keyword-based simulation. In a real-world scenario,
        # this would integrate with an actual NLP model or a cloud-based
        # sentiment analysis service (e.g., Hugging Face, Azure Cognitive Services, AWS Comprehend).
        positive_words = {"good", "great", "wonderful", "amazing", "love",
                          "excellent", "happy", "positive", "awesome", "fantastic",
                          "brilliant", "joy", "superb"}
        negative_words = {"bad", "terrible", "horrible", "awful", "hate",
                          "sad", "negative", "poor", "disappointing", "frustrating",
                          "unacceptable", "furious", "stress"}

        text_lower = text_to_analyze.lower()
        score = 0
        matched_positives = []
        matched_negatives = []

        # Count positive keywords
        for word in positive_words:
            if word in text_lower:
                count = text_lower.count(word)
                score += count
                matched_positives.append(f"'{word}' ({count})")

        # Count negative keywords
        for word in negative_words:
            if word in text_lower:
                count = text_lower.count(word)
                score -= count
                matched_negatives.append(f"'{word}' ({count})")

        # Determine sentiment label based on score
        if score > 0:
            sentiment_label = "positive"
        elif score < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        details = (
            f"Matched positives: {', '.join(matched_positives) if matched_positives else 'None'}. "
            f"Matched negatives: {', '.join(matched_negatives) if matched_negatives else 'None'}."
        )

        logger.info(
            f"[{self.node_name}] Successfully analyzed text: '{text_to_analyze[:50]}...'. "
            f"Result: Label='{sentiment_label}', Score={score}."
        )

        # --- Return Analysis Results ---
        return {
            'original_input': data, # Return original data for traceability
            'sentiment_analysis': {
                'label': sentiment_label,
                'score': float(score), # Ensure score is a float
                'details': details
            }
        }