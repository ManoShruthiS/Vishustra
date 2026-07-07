import logging
import re
from typing import Any, Dict, Union, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact personally identifiable information (PII)
    from input data using a predefined set of regular expressions.

    This node is capable of processing strings, as well as traversing string values within
    dictionaries and lists to identify and replace PII. Common PII patterns such as
    email addresses, phone numbers, and social security numbers are targeted and replaced
    with a standardized '[REDACTED]' placeholder.
    """

    # Define regular expression patterns for common PII types
    _PII_PATTERNS = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone_number": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b',
        # Future expansion: Add more sophisticated patterns for other PII types
    }
    _REPLACEMENT_TEXT = '[REDACTED]'

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this PII Redactor node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all defined PII redaction patterns to a given string,
        replacing identified PII with a placeholder.

        Args:
            text (str): The input string to be redacted.

        Returns:
            str: The string with all identified PII redacted.
        """
        redacted_text = text
        for pii_type, pattern in self._PII_PATTERNS.items():
            try:
                # Use re.sub to replace all occurrences of the pattern
                redacted_text = re.sub(pattern, self._REPLACEMENT_TEXT, redacted_text)
            except re.error as e:
                logger.error(f"[{self.node_name}] Failed to apply regex pattern for '{pii_type}': {e}")
                # Log the error but continue processing with other patterns
            except TypeError as e:
                logger.error(f"[{self.node_name}] Regex substitution failed with TypeError for '{pii_type}': {e}")
                # This typically means pattern or replacement is not a string, which should not happen here.
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the PII redaction process on the input data.

        This method intelligently handles various data types:
        - If `data` is a string, it directly applies PII redaction.
        - If `data` is a dictionary, it recursively processes string values and nested
          dictionaries/lists within it.
        - If `data` is a list, it recursively processes string elements and nested
          dictionaries/lists within it.
        - For any other data type, it logs a warning and returns the data unchanged,
          as redaction is typically not applicable or would require specific handling
          beyond the scope of this general PII node.

        Args:
            data (Any): The input data that may contain PII. This can be a string,
                        dictionary, list, or other types.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing task. Currently, this node
                                      does not use context for configuration, but it's
                                      available for future extensions (e.g., dynamic
                                      PII patterns based on context).

        Returns:
            Any: The processed data with identified PII redacted. The data structure
                 is preserved.
        """
        logger.debug(f"[{self.node_name}] Starting PII redaction for input data type: {type(data)}")

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                # Recursively process values that might contain PII
                redacted_data[key] = self.process(value, context)
            return redacted_data
        elif isinstance(data, list):
            redacted_data = []
            for item in data:
                # Recursively process items that might contain PII
                redacted_data.append(self.process(item, context))
            return redacted_data
        else:
            # Handle non-string, non-dict, non-list types.
            logger.warning(
                f"[{self.node_name}] Data type '{type(data).__name__}' is not supported for "
                "PII redaction. Returning data unchanged."
            )
            return data