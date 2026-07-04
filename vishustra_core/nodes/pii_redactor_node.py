import re
import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Pre-compiled PII redaction patterns and their corresponding replacement strings.
# For a production system, these patterns would be highly configurable,
# potentially loaded from a configuration service, and could be more sophisticated
# using NLP techniques for named entity recognition for more ambiguous PII like names.
PII_PATTERNS = {
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'): '[REDACTED_EMAIL]',
    re.compile(r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'): '[REDACTED_PHONE]',
    re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'): '[REDACTED_SSN]',  # Social Security Number (e.g., XXX-XX-XXXX)
    re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b'): '[REDACTED_CREDIT_CARD]', # Common Credit Card patterns (broad)
    re.compile(r'\b\d{5}(?:[-\s]\d{4})?\b'): '[REDACTED_ZIP_CODE]', # US Zip Code (e.g., 12345 or 12345-6789)
    re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'): '[REDACTED_IP_ADDRESS]', # IPv4 Address
}

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts common PII (Personally Identifiable Information)
    patterns from string data. It can process raw strings, or recursively redact
    strings found within dictionaries and lists.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all defined PII redaction patterns to a single input string.
        """
        redacted_text = text
        for pattern, replacement in PII_PATTERNS.items():
            redacted_text = pattern.sub(replacement, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII patterns.
        
        This method handles different data types:
        - If `data` is a string, it applies PII redaction directly.
        - If `data` is a dictionary or a list, it recursively calls itself
          to process string elements within those structures.
        - For other data types (e.g., numbers, booleans, None), it returns
          the data unchanged.

        Args:
            data: The input data, which can be a string, dict, list, or other types.
            context: A dictionary containing contextual information for the processing.
                     Not directly used for redaction logic in this node, but passed
                     through for consistency with the BaseNode interface and recursive calls.

        Returns:
            The processed data with PII redacted. The data type will match the input.
        """
        if data is None:
            logger.debug(f"[{self.node_name}] Received None data, returning as is.")
            return None

        if isinstance(data, str):
            logger.debug(f"[{self.node_name}] Processing string data for PII redaction.")
            try:
                return self._redact_string(data)
            except Exception as e:
                # Log the error but return original data to prevent data loss
                logger.error(f"[{self.node_name}] Error redacting string data: {e}", exc_info=True)
                return data

        elif isinstance(data, dict):
            logger.debug(f"[{self.node_name}] Processing dictionary data recursively for PII redaction.")
            redacted_dict = {}
            for key, value in data.items():
                redacted_dict[key] = self.process(value, context)  # Recursive call
            return redacted_dict

        elif isinstance(data, list):
            logger.debug(f"[{self.node_name}] Processing list data recursively for PII redaction.")
            redacted_list = []
            for item in data:
                redacted_list.append(self.process(item, context))  # Recursive call
            return redacted_list

        else:
            logger.debug(
                f"[{self.node_name}] Received non-string/dict/list data type ({type(data).__name__}), returning as is."
            )
            return data