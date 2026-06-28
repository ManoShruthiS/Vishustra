
import logging
import re
from typing import Any, Dict, List, Union

# CRITICAL: This import path is specified in the requirements.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node that filters profanity from text data.

    This node identifies and replaces predefined profane words within input text
    with a placeholder string (e.g., '***'). It supports processing single strings,
    lists of strings, and dictionaries where text content is typically stored
    under common keys like 'text', 'content', or 'message'.
    """
    
    _PROFANITIES: List[str] = [
        "fuck", "shit", "bitch", "asshole", "damn", "cunt", "bastard", "piss", "motherfucker", "fucker"
    ]
    """
    A list of profanity words to be filtered. These are case-insensitively matched.
    """
    
    _REPLACEMENT_STRING: str = "***"
    """
    The string used to replace identified profanities.
    """
    
    # Compile regex patterns for profanities only once at class loading to optimize performance.
    # This list ensures that patterns are compiled only a single time across all instances.
    _PROFANITY_PATTERNS: List[re.Pattern] = []
    if not _PROFANITY_PATTERNS: # Ensure compilation happens only once globally for the class
        for word in _PROFANITIES:
            # Using \b for word boundaries to prevent partial matches (e.g., "scunthorpe" -> "scun***orpe")
            # re.escape is used to handle any special regex characters that might appear in a profanity word
            _PROFANITY_PATTERNS.append(
                re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            )

    def __init__(self):
        """
        Initializes the ProfanityFilterNode.
        """
        logger.debug(f"'{self.node_name}' node initialized. Loaded {len(self._PROFANITIES)} profanity patterns.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ProfanityFilterNode"

    def _filter_text(self, text: str) -> str:
        """
        Applies the profanity filter to a single string.

        Args:
            text (str): The input string to filter.

        Returns:
            str: The filtered string with profanities replaced by `_REPLACEMENT_STRING`.
        """
        if not isinstance(text, str):
            logger.warning(
                f"'{self.node_name}': Expected string for filtering, but received type '{type(text).__name__}'. "
                "Returning original data without filtering."
            )
            return text

        filtered_text = text
        for pattern in self._PROFANITY_PATTERNS:
            filtered_text = pattern.sub(self._REPLACEMENT_STRING, filtered_text)
        return filtered_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to filter out profanity.

        This method supports:
        - `str`: Filters the string directly.
        - `list[str]`: Iterates through the list, filtering each string element.
                       Non-string elements in the list are returned as is.
        - `dict`: Scans for common text keys (e.g., 'text', 'content', 'message', 'prompt')
                  and filters their string values. Other dictionary keys and non-string
                  values are left untouched.

        Args:
            data (Any): The input data to be processed. Expected types are `str`, `list`, or `dict`.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing pipeline. Not directly used by this node,
                                     but passed along as per `BaseNode` contract.

        Returns:
            Any: The processed data with profanity filtered. If the input data type is
                 unsupported or an error occurs, the original data is returned to
                 maintain pipeline integrity.
        """
        logger.debug(f"'{self.node_name}': Starting processing for data of type '{type(data).__name__}'.")

        try:
            if isinstance(data, str):
                return self._filter_text(data)
            elif isinstance(data, list):
                # Process each item in the list if it's a string, otherwise keep as is
                return [
                    self._filter_text(item) if isinstance(item, str) else item
                    for item in data
                ]
            elif isinstance(data, dict):
                # Define common keys where text content might be found in a dictionary
                text_keys = ['text', 'content', 'message', 'prompt', 'query', 'description']
                modified_data = data.copy() # Work on a copy to avoid side effects on original input
                processed_any_key = False

                for key in text_keys:
                    if key in modified_data and isinstance(modified_data[key], str):
                        original_value = modified_data[key]
                        modified_data[key] = self._filter_text(original_value)
                        # Log if an actual change occurred
                        if modified_data[key] != original_value:
                            logger.debug(f"'{self.node_name}': Filtered content for key '{key}' in dictionary data.")
                        processed_any_key = True
                
                if not processed_any_key:
                    logger.warning(
                        f"'{self.node_name}': Dictionary data received, but no recognized text key "
                        f"('{', '.join(text_keys)}') with a string value was found. Returning original dictionary."
                    )
                return modified_data
            else:
                logger.warning(
                    f"'{self.node_name}': Unsupported data type '{type(data).__name__}'. "
                    "This node is designed for string, list[str], or dict. Returning data as is."
                )
                return data
        except Exception as e:
            # Catching broad exceptions to ensure robustness and prevent pipeline failures
            logger.error(
                f"'{self.node_name}': An unexpected error occurred during processing: {e}",
                exc_info=True # Include traceback in the log for detailed debugging
            )
            # In case of an error, return the original data. This prevents the pipeline
            # from crashing and allows downstream nodes to potentially handle the original content,
            # though it means the profanity filter was bypassed. A more strict policy might re-raise.
            return data

