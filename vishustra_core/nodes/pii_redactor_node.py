import re
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts common Personally Identifiable Information (PII)
    from string data.
    
    This node identifies patterns for emails, phone numbers, social security numbers,
    and IP addresses, replacing them with a standardized `[REDACTED]` placeholder.
    """

    _PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_number": r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
        "ssn": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",  # Matches XXX-XX-XXXX or XXX XX XXXX
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    }
    _REDACTION_PLACEHOLDER = "[REDACTED]"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII patterns.

        Args:
            data: The input data, expected to be a string. If not a string,
                  it will be returned unchanged after logging a warning.
            context: A dictionary containing contextual information for processing.
                     Currently not used for configuration, but available for future
                     extensions (e.g., custom PII patterns, redaction strategies).

        Returns:
            The data with identified PII redacted, or the original data if it's
            not a string or no PII was found.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string (type: {type(data).__name__}). "
                "PII redaction will be skipped. Returning original data."
            )
            return data

        redacted_data = str(data) # Ensure we work with a mutable copy if data was something like `str` subclass

        logger.debug(f"[{self.node_name}] Starting PII redaction process.")

        for pii_type, pattern in self._PII_PATTERNS.items():
            try:
                # Use re.sub to find and replace all occurrences of the pattern
                # Case-insensitive for some patterns might be useful, but standard PII
                # is usually case-sensitive enough for these basic patterns.
                redacted_data, num_substitutions = re.subn(
                    pattern, self._REDACTION_PLACEHOLDER, redacted_data, flags=re.IGNORECASE
                )
                if num_substitutions > 0:
                    logger.info(
                        f"[{self.node_name}] Redacted {num_substitutions} instances of {pii_type}."
                    )
            except re.error as e:
                logger.error(
                    f"[{self.node_name}] Regex error for PII type '{pii_type}' with pattern "
                    f"'{pattern}': {e}. Skipping this PII type."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred while processing "
                    f"PII type '{pii_type}': {e}. Skipping this PII type."
                )

        logger.debug(f"[{self.node_name}] PII redaction process completed.")
        return redacted_data

if __name__ == "__main__":
    # This block is for demonstrating and testing the node independently.
    # In a real Vishustra setup, nodes are orchestrated by the framework.
    logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbose output

    pii_node = PIIRedactorNode()

    test_data_1 = "My email is user@example.com, phone is +1-555-123-4567, and SSN is 123-45-6789. Also check 192.168.1.1."
    test_data_2 = "No PII here, just some random text."
    test_data_3 = 12345  # Non-string data
    test_data_4 = "Another email: test@domain.org and a number (987) 654-3210. IP: 10.0.0.255"
    test_data_5 = "Mixed PII: email@mail.com, 111-22-3333, (212) 555-1234. Repeated: email@mail.com"

    print(f"\n--- Testing {pii_node.node_name} ---")

    print("\nOriginal 1:", test_data_1)
    redacted_1 = pii_node.process(test_data_1, {})
    print("Redacted 1:", redacted_1)
    assert redacted_1 == "My email is [REDACTED], phone is [REDACTED], and SSN is [REDACTED]. Also check [REDACTED]."

    print("\nOriginal 2:", test_data_2)
    redacted_2 = pii_node.process(test_data_2, {})
    print("Redacted 2:", redacted_2)
    assert redacted_2 == "No PII here, just some random text."

    print("\nOriginal 3:", test_data_3)
    redacted_3 = pii_node.process(test_data_3, {})
    print("Redacted 3:", redacted_3)
    assert redacted_3 == test_data_3 # Should return original non-string data

    print("\nOriginal 4:", test_data_4)
    redacted_4 = pii_node.process(test_data_4, {})
    print("Redacted 4:", redacted_4)
    assert redacted_4 == "Another email: [REDACTED] and a number [REDACTED]. IP: [REDACTED]"

    print("\nOriginal 5:", test_data_5)
    redacted_5 = pii_node.process(test_data_5, {})
    print("Redacted 5:", redacted_5)
    assert redacted_5 == "Mixed PII: [REDACTED], [REDACTED], [REDACTED]. Repeated: [REDACTED]"

    print("\n--- All tests passed ---")
