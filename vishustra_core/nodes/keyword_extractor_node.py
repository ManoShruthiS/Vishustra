import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that simulates keyword extraction from text data.

    This node takes a string as input, processes it by converting to lowercase,
    removing punctuation, tokenizing into words, and filtering out a predefined
    set of stop words and very short words to produce a list of simulated keywords.
    """

    def __init__(self):
        """
        Initializes the KeywordExtractorNode with a basic set of stop words
        for simulation purposes.
        """
        # A simple, hardcoded set of stop words. In a production environment,
        # this would typically be loaded from an external resource or an NLP library.
        self._stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
            'for', 'nor', 'on', 'at', 'to', 'from', 'of', 'in', 'with', 'by',
            'this', 'that', 'it', 'its', 'he', 'she', 'they', 'we', 'you',
            'i', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'our', 'their',
            'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'not', 'no',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now', 'what', 'where', 'when',
            'who', 'whom', 'why', 'how'
        }
        logger.debug("KeywordExtractorNode initialized with a predefined set of stop words.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract simulated keywords.

        The method expects 'data' to be a string. It converts the text to lowercase,
        removes common punctuation, splits it into individual words, and then
        filters these words based on a list of internal stop words and minimum length.
        The `context` dictionary is provided for future extensibility but is not
        utilized by this specific node.

        Args:
            data: The input data, expected to be a string containing text to analyze.
            context: A dictionary containing additional contextual information for processing.
                     This node does not currently utilize the context.

        Returns:
            A list of unique strings representing the extracted keywords. The order
            of keywords is preserved from their first appearance in the processed text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during the keyword extraction process.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for KeywordExtractorNode. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"KeywordExtractorNode requires 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.info("Received an empty or whitespace-only string for keyword extraction. Returning an empty list.")
            return []

        try:
            # Convert text to lowercase to standardize
            processed_text = data.lower()

            # Remove punctuation. Using a regex to remove all non-alphanumeric
            # characters while keeping spaces to separate words.
            processed_text = re.sub(r'[^\w\s]', '', processed_text)

            # Split the text into individual words
            words = processed_text.split()

            extracted_keywords = []
            for word in words:
                # Filter out stop words and words that are too short to be meaningful
                if word not in self._stop_words and len(word) > 2:
                    extracted_keywords.append(word)

            # Return unique keywords while preserving their order of appearance
            # using dict.fromkeys (Python 3.7+ guarantees insertion order).
            unique_keywords = list(dict.fromkeys(extracted_keywords))

            logger.info(f"Successfully extracted {len(unique_keywords)} keywords from the input data.")
            logger.debug(f"Extracted keywords: {unique_keywords}")

            return unique_keywords

        except Exception as e:
            logger.exception(f"An unexpected error occurred during keyword extraction: {e}")
            raise ValueError(f"Failed to extract keywords due to an internal processing error: {e}") from e