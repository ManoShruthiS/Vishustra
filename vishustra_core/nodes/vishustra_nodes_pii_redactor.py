import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts common Personally Identifiable Information (PII)
    from string data or string values within dictionaries.

    Currently supports redaction of:
    - Email addresses
    - Phone numbers (common US/international formats)
    - IP addresses

    The redaction placeholders are configurable via node initialization or
    can be extended to be passed via the `context`.
    """

    def __init__(self,
                 email_placeholder: str = "[REDACTED_EMAIL]",
                 phone_placeholder: str = "[REDACTED_PHONE]",
                 ip_placeholder: str = "[REDACTED_IP]"):
        """
        Initializes the PII Redactor Node with custom placeholders.

        Args:
            email_placeholder (str): The string to replace email addresses with.
            phone_placeholder (str): The string to replace phone numbers with.
            ip_placeholder (str): The string to replace IP addresses with.
        """
        self._email_placeholder = email_placeholder
        self._phone_placeholder = phone_placeholder
        self._ip_placeholder = ip_placeholder

        # Compile regex patterns for efficiency
        self._patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "ip": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII_Redactor"

    def _redact_string(self, text: str) -> str:
        """Applies all redaction patterns to a given string."""
        original_text = text
        redacted_count = 0

        # Redact emails
        text, count = self._patterns["email"].subn(self._email_placeholder, text)
        redacted_count += count

        # Redact phone numbers
        text, count = self._patterns["phone"].subn(self._phone_placeholder, text)
        redacted_count += count

        # Redact IP addresses
        text, count = self._patterns["ip"].subn(self._ip_placeholder, text)
        redacted_count += count

        if redacted_count > 0:
            logger.debug("Redacted %d PII instances in string data.", redacted_count)
        else:
            logger.debug("No PII found for redaction in string data.")

        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        If `data` is a string, it redacts PII directly within the string.
        If `data` is a dictionary, it iterates through its top-level string values
        and redacts PII within them, returning a new dictionary.
        Other data types are returned unchanged with a warning.

        Args:
            data (Any): The input data, expected to be a string or a dictionary
                        containing string values.
            context (Dict[str, Any]): A dictionary providing additional context
                                     (not directly used for configuration here,
                                     but available for future extensions like
                                     dynamic pattern loading).

        Returns:
            Any: The processed data with PII redacted, or the original data
                 if it's not a supported type.
        """
        if isinstance(data, str):
            try:
                redacted_data = self._redact_string(data)
                logger.info("String data processed for PII redaction.")
                return redacted_data
            except Exception as e:
                logger.exception("Error during PII redaction of string data: %s", e)
                raise # Re-raise to signal processing failure

        elif isinstance(data, dict):
            redacted_dict = {}
            for key, value in data.items():
                if isinstance(value, str):
                    try:
                        redacted_dict[key] = self._redact_string(value)
                    except Exception as e:
                        logger.error("Error redacting PII in dict key '%s': %s. Value left unchanged.", key, e)
                        redacted_dict[key] = value # Keep original value on error
                else:
                    redacted_dict[key] = value # Non-string values are passed through
            logger.info("Dictionary data processed for PII redaction.")
            return redacted_dict

        else:
            logger.warning(
                "Unsupported data type '%s' for PII redaction. Data returned unchanged.",
                type(data).__name__
            )
            return data

# Example of how to use (for testing purposes, not part of the required output)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    redactor = PIIRedactorNode()

    # Test with string data
    test_string_1 = "My email is user@example.com and my phone is +1-555-123-4567. The server IP is 192.168.1.1."
    redacted_string_1 = redactor.process(test_string_1, {})
    logger.info("Original string: %s", test_string_1)
    logger.info("Redacted string: %s", redacted_string_1)
    # Expected: "My email is [REDACTED_EMAIL] and my phone is [REDACTED_PHONE]. The server IP is [REDACTED_IP]."

    test_string_2 = "No PII here, just some text."
    redacted_string_2 = redactor.process(test_string_2, {})
    logger.info("Original string: %s", test_string_2)
    logger.info("Redacted string: %s", redacted_string_2)
    # Expected: "No PII here, just some text."

    # Test with dictionary data
    test_dict_1 = {
        "name": "John Doe",
        "contact_email": "john.doe@company.org",
        "message": "Please call me at (123) 456-7890. My IP is 10.0.0.5.",
        "user_id": 12345
    }
    redacted_dict_1 = redactor.process(test_dict_1, {})
    logger.info("Original dict: %s", test_dict_1)
    logger.info("Redacted dict: %s", redacted_dict_1)
    # Expected: {'name': 'John Doe', 'contact_email': '[REDACTED_EMAIL]', 'message': 'Please call me at [REDACTED_PHONE]. My IP is [REDACTED_IP].', 'user_id': 12345}

    test_dict_2 = {
        "event": "login_attempt",
        "timestamp": "2023-10-27T10:00:00Z"
    }
    redacted_dict_2 = redactor.process(test_dict_2, {})
    logger.info("Original dict: %s", test_dict_2)
    logger.info("Redacted dict: %s", redacted_dict_2)
    # Expected: {'event': 'login_attempt', 'timestamp': '2023-10-27T10:00:00Z'}

    # Test with unsupported data type
    test_list = ["email@test.com", "123-456-7890"]
    redacted_list = redactor.process(test_list, {})
    logger.info("Original list (unsupported): %s", test_list)
    logger.info("Redacted list (should be same): %s", redacted_list)
    # Expected: ['email@test.com', '123-456-7890'] (and a warning log)

    # Test with custom placeholders
    custom_redactor = PIIRedactorNode(
        email_placeholder="<EMAIL_HIDDEN>",
        phone_placeholder="<PHONE_NUMBER_HIDDEN>",
        ip_placeholder="<IP_ADDRESS_HIDDEN>"
    )
    test_string_3 = "Email: user@domain.net, Call: 987.654.3210, From IP: 203.0.113.10"
    redacted_string_3 = custom_redactor.process(test_string_3, {})
    logger.info("Original string (custom placeholders): %s", test_string_3)
    logger.info("Redacted string (custom placeholders): %s", redacted_string_3)
    # Expected: "Email: <EMAIL_HIDDEN>, Call: <PHONE_NUMBER_HIDDEN>, From IP: <IP_ADDRESS_HIDDEN>"