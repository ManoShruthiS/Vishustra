import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and redact Personally Identifiable Information (PII)
    from text data. This node uses regular expressions to detect common PII patterns
    such as email addresses, phone numbers, and social security numbers.
    """

    def __init__(self, redaction_string: str = "[REDACTED_PII]") -> None:
        """
        Initializes the PIIRedactorNode with a specified redaction string.

        Args:
            redaction_string (str): The string to replace detected PII with.
                                     Defaults to "[REDACTED_PII]".
        """
        self._redaction_string = redaction_string
        # Define common PII patterns with corresponding keys for logging
        self._pii_patterns = {
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
            "phone_number": re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), # US-like Social Security Number
            "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            # Add more patterns as needed (e.g., credit card numbers, names - more complex)
        }
        logger.debug(f"PIIRedactorNode initialized with redaction string: '{self._redaction_string}'")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used for redaction
                                       in this implementation but available for future extensions.

        Returns:
            Any: The processed data with PII redacted, or the original data
                 if it's not a string or an error occurs during processing.

        Raises:
            TypeError: If the input data is not a string. (Decided against raising for robustness)
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string (type: {type(data).__name__}). "
                "PII redaction will be skipped. Returning original data."
            )
            return data

        redacted_data = data
        pii_found_count = 0

        logger.info(f"[{self.node_name}] Starting PII redaction for input data.")

        for pii_type, pattern in self._pii_patterns.items():
            try:
                matches = pattern.findall(redacted_data)
                if matches:
                    unique_matches = set(matches) # Redact each unique instance once
                    for match in unique_matches:
                        # Use re.sub with a limit of 1 to replace one instance at a time to count accurately
                        # Or, simply replace all at once with a single sub call for efficiency
                        redacted_data = pattern.sub(self._redaction_string, redacted_data)
                        pii_found_count += len(matches) # Count all occurrences

                    logger.debug(
                        f"[{self.node_name}] Redacted {len(unique_matches)} unique instance(s) of '{pii_type}'."
                    )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Error during redaction of '{pii_type}' pattern: {e}"
                )
                # Continue with other patterns or decide to re-raise based on error severity

        if pii_found_count > 0:
            logger.info(
                f"[{self.node_name}] Completed PII redaction. Total PII instances found and "
                f"redacted across all patterns: {pii_found_count}."
            )
        else:
            logger.info(f"[{self.node_name}] No PII found in the input data.")

        return redacted_data

# Example of how to use this node (for testing purposes, not part of the final commit)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock the BaseNode for local testing without the full framework structure
    # In a real scenario, vishustra_core.nodes.base_node would be available.
    if 'BaseNode' not in globals():
        from abc import ABC, abstractmethod
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass

    # Re-define the node with the local BaseNode for testing
    class PIIRedactorNode(BaseNode):
        def __init__(self, redaction_string: str = "[REDACTED_PII]") -> None:
            self._redaction_string = redaction_string
            self._pii_patterns = {
                "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
                "phone_number": re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
                "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            }
            logger.debug(f"PIIRedactorNode initialized with redaction string: '{self._redaction_string}'")

        @property
        def node_name(self) -> str:
            return "PIIRedactorNode"

        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            if not isinstance(data, str):
                logger.warning(
                    f"[{self.node_name}] Input data is not a string (type: {type(data).__name__}). "
                    "PII redaction will be skipped. Returning original data."
                )
                return data

            redacted_data = data
            pii_found_count = 0

            logger.info(f"[{self.node_name}] Starting PII redaction for input data.")

            for pii_type, pattern in self._pii_patterns.items():
                try:
                    matches = pattern.findall(redacted_data)
                    if matches:
                        redacted_data = pattern.sub(self._redaction_string, redacted_data)
                        pii_found_count += len(matches)
                        logger.debug(f"[{self.node_name}] Redacted {len(matches)} instance(s) of '{pii_type}'.")
                except Exception as e:
                    logger.error(f"[{self.node_name}] Error during redaction of '{pii_type}' pattern: {e}")

            if pii_found_count > 0:
                logger.info(
                    f"[{self.node_name}] Completed PII redaction. Total PII instances found and "
                    f"redacted across all patterns: {pii_found_count}."
                )
            else:
                logger.info(f"[{self.node_name}] No PII found in the input data.")

            return redacted_data


    redactor = PIIRedactorNode()

    test_data_1 = "My email is user@example.com, phone is 123-456-7890, and SSN is 999-88-7777. My IP is 192.168.1.1."
    test_data_2 = "No sensitive info here."
    test_data_3 = "Another email: test.user@domain.co.uk and number +1 (555) 123-4567. " \
                  "IPs: 10.0.0.1 and 172.16.0.2."
    test_data_4 = 12345 # Non-string data

    print("\n--- Test Case 1 ---")
    print("Original:", test_data_1)
    print("Redacted:", redactor.process(test_data_1, {}))

    print("\n--- Test Case 2 ---")
    print("Original:", test_data_2)
    print("Redacted:", redactor.process(test_data_2, {}))

    print("\n--- Test Case 3 ---")
    print("Original:", test_data_3)
    print("Redacted:", redactor.process(test_data_3, {}))

    print("\n--- Test Case 4 (Non-string) ---")
    print("Original:", test_data_4)
    print("Redacted:", redactor.process(test_data_4, {}))

    print("\n--- Test with custom redaction string ---")
    custom_redactor = PIIRedactorNode(redaction_string="[ANONYMIZED]")
    print("Original:", test_data_1)
    print("Redacted:", custom_redactor.process(test_data_1, {}))