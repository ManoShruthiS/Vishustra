import re
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for identifying and redacting
    personally identifiable information (PII) from textual data.

    This node uses regular expressions to detect common PII patterns such as
    email addresses and phone numbers. It can operate on strings, or recursively
    on strings within dictionaries and lists.
    """

    DEFAULT_REDACTION_STRING: str = "[REDACTED_PII]"

    def __init__(self, redaction_string: str = DEFAULT_REDACTION_STRING):
        """
        Initializes the PIIRedactorNode with a specified redaction string.

        Args:
            redaction_string: The string to replace identified PII with.
                              Defaults to "[REDACTED_PII]".
        """
        self._redaction_string = redaction_string
        # Define common PII patterns using regex
        # Using a tuple of (pattern, description) for better logging/future expansion
        self._pii_patterns: List[re.Pattern] = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), # Email
            re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), # Common phone numbers (US/International)
        ]
        logger.info(f"PIIRedactorNode initialized with redaction string: '{self._redaction_string}'")

    @property
    def node_name(self) -> str:
        """
        Returns the name of the node.
        """
        return "PII_Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction patterns to a given string.
        """
        redacted_text = text
        for pattern in self._pii_patterns:
            matches = list(pattern.finditer(redacted_text))
            if matches:
                logger.debug(f"Found PII matches for pattern '{pattern.pattern}' in string.")
                redacted_text = pattern.sub(self._redaction_string, redacted_text)
        return redacted_text

    def _process_recursive(self, data: Any) -> Any:
        """
        Recursively processes data to redact PII within strings, dictionaries, and lists.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._process_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_recursive(item) for item in data]
        else:
            # For other types, return as is. Log if it's not a common primitive.
            if not isinstance(data, (int, float, bool, type(None))):
                logger.debug(f"Skipping PII redaction for unsupported data type: {type(data)}")
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        The method handles string inputs directly. For dictionary or list inputs,
        it traverses them recursively to find and redact PII within nested strings.
        Other data types are returned unchanged.

        Args:
            data: The input data, which can be a string, dict, list, or other types.
            context: A dictionary containing contextual information for the processing.
                     (Currently not used for configuration in this node, but available).

        Returns:
            The processed data with PII redacted.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process.")
        logger.debug(f"[{self.node_name}] Context received: {context}")

        try:
            redacted_data = self._process_recursive(data)
            logger.info(f"[{self.node_name}] PII redaction completed successfully.")
            return redacted_data
        except Exception as e:
            logger.error(f"[{self.node_name}] Error during PII redaction: {e}", exc_info=True)
            # Depending on policy, might re-raise or return original data.
            # For robust frameworks, often prefer to return original on error or raise custom exception.
            # Here, we return original data and log the error.
            return data

if __name__ == '__main__':
    # Example Usage and Testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    redactor = PIIRedactorNode()

    # Test cases
    test_data_string = "My email is test@example.com and phone is 123-456-7890. Also, call me at (987)654-3210. Another email: user@domain.org."
    test_data_dict = {
        "user_info": {
            "name": "John Doe",
            "email": "john.doe@email.co.uk",
            "phone": "+1 555 123 4567",
            "address": "123 PII St."
        },
        "message": "Please contact support@company.com for assistance.",
        "id": 12345
    }
    test_data_list = [
        "First item, no PII.",
        {"contact": "info@service.net"},
        "My number is 555 111 2222."
    ]
    test_data_mixed = {
        "documents": [
            {"text": "Email: alice@example.info, Phone: 01-2345-6789"},
            "Random text with no PII."
        ],
        "sender": "sender@mail.com",
        "metadata": {"timestamp": "2023-10-27T10:00:00Z"}
    }
    test_data_unsupported = 123456789

    print("\n--- Test Case 1: String Input ---")
    redacted_string = redactor.process(test_data_string, {})
    print(f"Original: {test_data_string}")
    print(f"Redacted: {redacted_string}")

    print("\n--- Test Case 2: Dictionary Input ---")
    redacted_dict = redactor.process(test_data_dict, {})
    print(f"Original: {test_data_dict}")
    print(f"Redacted: {redacted_dict}")

    print("\n--- Test Case 3: List Input ---")
    redacted_list = redactor.process(test_data_list, {})
    print(f"Original: {test_data_list}")
    print(f"Redacted: {redacted_list}")

    print("\n--- Test Case 4: Mixed Input (Dict containing List) ---")
    redacted_mixed = redactor.process(test_data_mixed, {})
    print(f"Original: {test_data_mixed}")
    print(f"Redacted: {redacted_mixed}")

    print("\n--- Test Case 5: Unsupported Type Input ---")
    redacted_unsupported = redactor.process(test_data_unsupported, {})
    print(f"Original: {test_data_unsupported} (Type: {type(test_data_unsupported)})")
    print(f"Redacted: {redacted_unsupported}")

    # Test with custom redaction string
    print("\n--- Test Case 6: Custom Redaction String ---")
    redactor_custom = PIIRedactorNode(redaction_string="[MASKED]")
    test_string_custom = "My contact is user@gmail.com or 111-222-3333."
    redacted_string_custom = redactor_custom.process(test_string_custom, {})
    print(f"Original: {test_string_custom}")
    print(f"Redacted: {redacted_string_custom}")