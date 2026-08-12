import logging
import re
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node that redacts personally identifiable information (PII)
    from text data, supporting various data structures.

    This node uses regular expressions to identify common PII patterns
    such as email addresses, phone numbers, and SSN-like patterns,
    replacing them with configurable placeholder tags.
    """

    def __init__(self,
                 pii_patterns: Dict[str, str] = None,
                 replacement_tags: Dict[str, str] = None,
                 default_replacement: str = "[REDACTED]"):
        """
        Initializes the PII Redactor node with custom patterns or default ones.

        Args:
            pii_patterns (Dict[str, str], optional): A dictionary where keys are PII types
                (e.g., "email", "phone") and values are their corresponding regex patterns.
                If None, a set of default patterns will be used.
            replacement_tags (Dict[str, str], optional): A dictionary where keys are PII types
                and values are the specific replacement strings for that type.
                If a PII type is not in this dict, `default_replacement` is used.
            default_replacement (str, optional): The default string to replace identified PII.
                Defaults to "[REDACTED]".
        """
        self.default_replacement = default_replacement
        self.pii_patterns = pii_patterns if pii_patterns is not None else self._get_default_pii_patterns()
        self.replacement_tags = replacement_tags if replacement_tags is not None else self._get_default_replacement_tags()
        logger.info(f"Initialized PII Redactor node with {len(self.pii_patterns)} patterns.")

    def _get_default_pii_patterns(self) -> Dict[str, str]:
        """Provides a set of default PII regex patterns."""
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', # Covers various formats
            "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b', # US Social Security Number format
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            # Add more patterns as needed, e.g., credit card numbers (careful with PCI compliance!)
        }

    def _get_default_replacement_tags(self) -> Dict[str, str]:
        """Provides default specific replacement tags for common PII types."""
        return {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]",
            "ssn_us": "[SSN_REDACTED]",
            "ip_address": "[IP_REDACTED]",
        }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def _redact_value(self, value: Any) -> Any:
        """
        Recursively redacts PII from strings within various data structures.
        """
        if isinstance(value, str):
            redacted_string = value
            for pii_type, pattern in self.pii_patterns.items():
                try:
                    replacement = self.replacement_tags.get(pii_type, self.default_replacement)
                    redacted_string = re.sub(pattern, replacement, redacted_string)
                except re.error as e:
                    logger.error(f"Regex error during PII redaction for type '{pii_type}': {e}")
                    # Continue with original string if regex fails for a specific pattern
                except Exception as e:
                    logger.error(f"An unexpected error occurred while redacting PII type '{pii_type}': {e}")
            return redacted_string
        elif isinstance(value, dict):
            return {k: self._redact_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._redact_value(item) for item in value]
        else:
            # If data type is not string, dict, list, or tuple, return as is.
            # E.g., numbers, booleans, None.
            return value

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The method handles various data types:
        - If `data` is a string, it directly applies PII redaction.
        - If `data` is a dictionary or list/tuple, it recursively processes
          string values within these structures.
        - For other data types, it returns the data unchanged.

        Args:
            data (Any): The input data to be processed for PII. Can be a string,
                        dictionary, list, or other types.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the current processing flow. Not directly
                                     used by this node for redaction logic but
                                     available for future extensions.

        Returns:
            Any: The data with identified PII redacted, or the original data
                 if no PII was found or if the data type is not processable.
        """
        logger.debug(f"[{self.node_name}] Starting PII redaction for incoming data.")
        try:
            redacted_data = self._redact_value(data)
            logger.debug(f"[{self.node_name}] PII redaction completed.")
            return redacted_data
        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to process data for PII redaction: {e}", exc_info=True)
            # Depending on policy, might return original data or raise the exception
            return data # Return original data in case of unexpected processing failure
