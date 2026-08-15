import logging
from typing import Any, Dict, Union, List, Tuple
import re

# Assume BaseNode is available in the specified path based on project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A processing node designed for basic content moderation by filtering out profanities.

    This node identifies a predefined set of common profanities within string data
    and replaces them with asterisks (*) of the same length, ensuring text length
    and structural integrity are maintained. It can process single strings, lists
    or tuples of strings, and dictionaries where values might be strings or nested
    collections.
    """

    # A private set of profanities for detection. In a production system, this list
    # would ideally be loaded from a configurable source (e.g., database, external file)
    # and potentially allow for user-defined rules or context-specific lists.
    _profanities: set[str] = {
        "fuck", "shit", "bitch", "asshole", "damn", "cunt", "motherfucker",
        "bastard", "dick", "pussy", "fag", # Note: 'gay' is omitted due to potential for misinterpretation
                                         # as a slur vs. a descriptive term. Careful list curation is key.
        "bollocks", "wanker", "prick", "sodoff", "arse", "bugger", "hell", # Adding common UK profanities
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilter"

    def _replace_profanities_in_string(self, text: str) -> str:
        """
        Helper method to replace profanities in a single string.

        This method tokenizes the input string by whitespace, punctuation, and
        attempts to replace whole-word profanities. It preserves the original
        casing and surrounding punctuation of the filtered words.
        """
        if not isinstance(text, str):
            logger.debug(f"Expected string for profanity filtering, got {type(text)}. Skipping.")
            return text

        # Use regex to find words, including words with leading/trailing punctuation
        # This allows matching "fuck." but replacing only "fuck" and keeping the ".".
        # A more robust solution might use an NLP tokenizer, but for simple simulation,
        # this regex split is reasonably effective.
        
        # Pattern to split text while keeping delimiters (punctuation/spaces)
        # This pattern splits by non-word characters but includes them in the result
        # allowing us to rebuild the string with original punctuation.
        parts = re.split(r'(\W+)', text)
        filtered_parts = []

        for part in parts:
            if not part.strip(): # Skip empty parts or pure whitespace
                filtered_parts.append(part)
                continue

            # Check if the part is a word that might contain profanity
            # Normalize for comparison by making lowercase and stripping common punctuation
            normalized_part = part.lower().strip(".,!?;:\"'()[]{}<>")

            if normalized_part in self._profanities:
                # Replace the original word part with asterisks, preserving original length
                # and any surrounding punctuation from the original part.
                # Example: "fuck!" becomes "****!"
                profanity_len = len(normalized_part)
                
                # Reconstruct the part: prefix, asterisks, suffix
                # Find the actual word within the part (e.g., from " (fuck!) " get "fuck")
                match = re.search(r'\b(' + re.escape(normalized_part) + r')\b', part, re.IGNORECASE)
                if match:
                    # Replace the exact matched word with asterisks
                    start, end = match.span()
                    filtered_part = part[:start] + ("*" * profanity_len) + part[end:]
                    filtered_parts.append(filtered_part)
                else:
                    # Fallback if regex doesn't find exact word within part
                    # (e.g., if punctuation causes issues, replace whole part)
                    filtered_parts.append("*" * len(part))
            else:
                filtered_parts.append(part)
        
        return "".join(filtered_parts)


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanities.

        This method intelligently handles various data structures:
        -   **Single strings**: Profanities are replaced with asterisks.
        -   **Lists/Tuples**: Iterates through the collection, filtering string elements.
            Non-string elements (including nested collections) are processed recursively.
        -   **Dictionaries**: Iterates through key-value pairs, filtering string values.
            Non-string values (including nested collections) are processed recursively.
        -   **Other types**: Are passed through unchanged, and a warning is logged.

        Args:
            data (Any): The input data to be processed. This can be a string,
                        a list/tuple of strings or nested structures, or a dictionary
                        with string values or nested structures.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. While not directly utilized
                                      by this specific node, it is passed along for
                                      conformity with the `BaseNode` interface.

        Returns:
            Any: The processed data with identified profanities filtered out.
                 If the data type is unsupported, the original data is returned
                 unchanged.
        """
        logger.debug(f"ProfanityFilterNode received data of type: {type(data).__name__}")

        if isinstance(data, str):
            return self._replace_profanities_in_string(data)
        elif isinstance(data, (list, tuple)):
            processed_data = [self.process(item, context) for item in data]
            # Return the processed data in its original collection type (list or tuple)
            return type(data)(processed_data)
        elif isinstance(data, dict):
            processed_data = {
                key: self.process(value, context) for key, value in data.items()
            }
            return processed_data
        else:
            logger.warning(
                f"ProfanityFilterNode received unsupported data type '{type(data).__name__}'. "
                "Returning data unchanged without applying filter."
            )
            return data