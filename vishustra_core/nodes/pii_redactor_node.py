import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and redact Personally Identifiable Information (PII)
    from incoming data. It supports redaction within strings, dictionaries, and lists of strings.
    """

    # Pre-compiled regex patterns for common PII types
    _PII_PATTERNS = {
        "email": {
            "regex": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "replacement": "[REDACTED_EMAIL]"
        },
        "phone_number": {
            "regex": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "replacement": "[REDACTED_PHONE]"
        },
        "ssn": {
            "regex": re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'), # Matches XXX-XX-XXXX, XXX XX XXXX, XXXXXXXXX
            "replacement": "[REDACTED_SSN]"
        },
        "credit_card": {
            "regex": re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){3}\d{3,4}\b'),
            "replacement": "[REDACTED_CREDIT_CARD]"
        },
        "ip_address": {
            "regex": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "replacement": "[REDACTED_IP]"
        }
        # Add more patterns as needed (e.g., specific names, addresses, dates of birth if context allows)
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """Applies all PII redaction patterns to a given string."""
        redacted_text = text
        for pii_type, pattern_info in self._PII_PATTERNS.items():
            regex = pattern_info["regex"]
            replacement = pattern_info["replacement"]
            matches = regex.findall(redacted_text)
            if matches:
                logger.debug(f"Found {len(matches)} PII matches of type '{pii_type}' in string.")
                redacted_text = regex.sub(replacement, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        The `data` can be:
        - A string: PII patterns are applied directly.
        - A dictionary: String values within the dictionary (at the top level)
                        are processed.
        - A list: Each element in the list is processed (if it's a string or dict).
        - Any other type: The data is returned unchanged.

        Args:
            data: The input data to be processed for PII redaction.
            context: A dictionary containing contextual information for the node.
                     Currently not used for PII patterns, but can be extended
                     to allow dynamic pattern configuration.

        Returns:
            The data with identified PII redacted.
        """
        logger.debug(f"Processing data with PII Redactor node. Data type: {type(data)}")

        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            redacted_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    redacted_data[key] = self._redact_string(value)
                elif isinstance(value, (dict, list)):
                    # Recursively process nested dictionaries and lists
                    redacted_data[key] = self.process(value, context)
                else:
                    redacted_data[key] = value
            return redacted_data
        elif isinstance(data, list):
            redacted_list = []
            for item in data:
                if isinstance(item, (str, dict, list)):
                    redacted_list.append(self.process(item, context))
                else:
                    redacted_list.append(item)
            return redacted_list
        else:
            logger.warning(
                f"Unsupported data type for PII redaction: {type(data)}. "
                "Returning data unchanged. Supported types are str, dict, list."
            )
            return data

# Example usage (for local testing/demonstration, not part of the node itself)
if __name__ == '__main__':
    # Configure logging for demonstration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Ensure this logger instance also logs DEBUG

    redactor = PIIRedactorNode()
    context = {} # Empty context for now

    print(f"\n--- Node Name: {redactor.node_name} ---")

    # Test cases
    test_cases = [
        "My email is alice.smith@example.com and phone is +1 (555) 123-4567. SSN: 123-45-6789.",
        "Another email: bob@test.org. IP address is 192.168.1.1. My credit card is 1234-5678-9012-3456.",
        {"user_id": "u123", "message": "Contact me at charlie@domain.net or call 555.987.6543.", "sensitive_info": "SSN: 987 65 4321"},
        {"profile": {"email": "diana@mail.com", "phone": "001-555-111-2222"}, "data": ["item1", "item2", "secret_email@hidden.org"]},
        ["hello world", "email@foo.bar", "My card: 4111 2222 3333 4444"],
        12345, # Non-string/dict/list data
        None,
        {"nested_list": ["item A", "email@list.com", {"deep_dict": "data with pii@deep.org"}]}
    ]

    for i, data in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Original: {data}")
        try:
            redacted_data = redactor.process(data, context)
            print(f"Redacted: {redacted_data}")
        except Exception as e:
            print(f"Error processing data: {e}")
