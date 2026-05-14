import re
import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A node responsible for identifying and redacting Personally Identifiable Information (PII)
    from input strings or dictionaries. Supports common patterns like emails, 
    phone numbers, and generic credit card formats.
    """

    def __init__(self, replacement_text: str = "[REDACTED]"):
        self._replacement_text = replacement_text
        # Common regex patterns for PII detection
        self._patterns = {
            "email": re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'),
            "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            "ipv4": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node."""
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Redacts PII from the provided data. 
        
        Args:
            data: The input data, expected to be a string or a dictionary.
            context: Execution context containing configuration or metadata.

        Returns:
            The processed data with sensitive information masked.
        """
        try:
            if isinstance(data, str):
                return self._redact_string(data)
            elif isinstance(data, dict):
                return self._redact_dict(data)
            elif isinstance(data, list):
                return [self.process(item, context) for item in data]
            else:
                logger.warning(f"[{self.node_name}] Received unsupported data type: {type(data)}. Returning as is.")
                return data
        except Exception as e:
            logger.error(f"[{self.node_name}] Error during PII redaction: {str(e)}", exc_info=True)
            raise ValueError(f"PII Redaction failed: {e}") from e

    def _redact_string(self, text: str) -> str:
        """Applies regex patterns to a string to mask PII."""
        redacted_text = text
        for pii_type, pattern in self._patterns.items():
            redacted_text = pattern.sub(self._replacement_text, redacted_text)
        return redacted_text

    def _redact_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively traverses a dictionary to redact string values."""
        new_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, str):
                new_dict[key] = self._redact_string(value)
            elif isinstance(value, dict):
                new_dict[key] = self._redact_dict(value)
            elif isinstance(value, list):
                new_dict[key] = [
                    self._redact_dict(item) if isinstance(item, dict) 
                    else (self._redact_string(item) if isinstance(item, str) else item)
                    for item in value
                ]
            else:
                new_dict[key] = value
        return new_dict