import re
import logging
from typing import Any, Dict, Union, List

# Assuming BaseNode is available at this path as per project instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PII_RedactorNode(BaseNode):
    """
    A Vishustra node designed to redact Personally Identifiable Information (PII)
    from text data. This node identifies common PII patterns using regular expressions
    and replaces them with generic '[REDACTED_TYPE]' placeholders.

    It supports redaction within:
    - Pure string inputs.
    - String values within dictionaries (including nested dictionaries and lists of strings/dictionaries).

    Supported PII types for redaction include:
    - Email addresses
    - Phone numbers (common international and North American formats)
    - Basic Credit Card Numbers (13-16 digits, with optional spaces/hyphens)
    - Social Security Numbers (US format: ###-##-####)
    - IP Addresses (IPv4)
    """

    # Compiled regular expressions for various PII patterns
    _email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    _phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[ -. ]?\d{3}[ -. ]?\d{4}\b')
    # This credit card pattern is a heuristic and might match non-CC numbers.
    # For high-assurance redaction, integrating Luhn algorithm validation would be necessary.
    _credit_card_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
    _ssn_pattern = re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b')
    _ip_address_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PII Redactor"

    def process(self, data: Union[str, Dict[str, Any], List[Any]], context: Dict[str, Any]) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Processes the input data to identify and redact PII.

        If the input `data` is a string, it redacts PII directly.
        If the input `data` is a dictionary or list, it recursively attempts to redact PII
        from all string values within the structure. Non-string values are returned unchanged.

        Args:
            data: The input data, expected to be a string, dictionary, or list containing
                  string values or nested structures.
            context: A dictionary containing contextual information relevant to the
                     orchestration, though not directly used for redaction logic in this node.

        Returns:
            The data with PII redacted. The return type matches the input type where possible.

        Raises:
            TypeError: If the input data is of an unsupported fundamental type.
            Exception: Catches and logs any unexpected errors during the redaction process.
        """
        if data is None:
            logger.debug("Received None data for PII redaction. Returning None.")
            return None

        try:
            if isinstance(data, str):
                return self._redact_string(data)
            elif isinstance(data, dict):
                return self._redact_dict(data)
            elif isinstance(data, list):
                return self._redact_list(data)
            else:
                error_msg = f"Unsupported data type for PII redaction: {type(data).__name__}. Expected str, dict, or list."
                logger.error(error_msg)
                raise TypeError(error_msg)
        except Exception as e:
            logger.exception(f"An unexpected error occurred during PII redaction: {e}")
            raise

    def _redact_string(self, text: str) -> str:
        """
        Helper method to apply PII redaction rules to a single string.
        """
        redacted_text = text
        
        redacted_text = self._email_pattern.sub('[REDACTED_EMAIL]', redacted_text)
        redacted_text = self._phone_pattern.sub('[REDACTED_PHONE]', redacted_text)
        redacted_text = self._credit_card_pattern.sub('[REDACTED_CC]', redacted_text)
        redacted_text = self._ssn_pattern.sub('[REDACTED_SSN]', redacted_text)
        redacted_text = self._ip_address_pattern.sub('[REDACTED_IP]', redacted_text)

        if text != redacted_text:
            logger.debug(f"PII redacted from string. Original length: {len(text)}, Redacted length: {len(redacted_text)}.")
        return redacted_text

    def _redact_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper method to recursively redact PII from string values within a dictionary.
        """
        redacted_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, str):
                redacted_dict[key] = self._redact_string(value)
            elif isinstance(value, dict):
                redacted_dict[key] = self._redact_dict(value)
            elif isinstance(value, list):
                redacted_dict[key] = self._redact_list(value)
            else:
                redacted_dict[key] = value # Keep non-string/non-dict/non-list values as is
        logger.debug(f"PII redaction applied to dictionary with {len(data_dict)} items.")
        return redacted_dict

    def _redact_list(self, data_list: List[Any]) -> List[Any]:
        """
        Helper method to recursively redact PII from string values within a list.
        """
        redacted_list = []
        for item in data_list:
            if isinstance(item, str):
                redacted_list.append(self._redact_string(item))
            elif isinstance(item, dict):
                redacted_list.append(self._redact_dict(item))
            elif isinstance(item, list):
                redacted_list.append(self._redact_list(item))
            else:
                redacted_list.append(item) # Keep non-string/non-dict/non-list items as is
        logger.debug(f"PII redaction applied to list with {len(data_list)} items.")
        return redacted_list