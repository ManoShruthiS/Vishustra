import re
import logging
from typing import Any, Dict

# Assuming BaseNode is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and redact personally identifiable
    information (PII) from text data. It uses regular expressions to detect
    common PII patterns such as email addresses, phone numbers, and IP addresses,
    replacing them with a configurable placeholder.
    """

    def __init__(self, redaction_placeholder: str = "[REDACTED]") -> None:
        """
        Initializes the PIIRedactorNode.

        Args:
            redaction_placeholder (str): The string used to replace identified PII.
                                         Defaults to "[REDACTED]".
        """
        self._redaction_placeholder = redaction_placeholder
        # Compile common PII regex patterns for efficiency.
        # These patterns are illustrative and can be extended based on specific needs.
        self._pii_patterns = {
            "email": re.compile(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                re.IGNORECASE
            ),
            "phone": re.compile(
                r"(?:\+?\d{1,3}[-. ]?)?\(?\d{2,4}\)?[-. ]?\d{2,4}[-. ]?\d{4}\b"
                r"|\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b" # Covers common US-like formats
            ),
            "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            # Additional patterns could include credit card numbers, national IDs, etc.,
            # which would require more robust and country-specific regex.
        }
        logger.debug(
            f"[{self.node_name}] Initialized with redaction placeholder: "
            f"'{self._redaction_placeholder}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PIIRedactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, redacting identified PII.

        This method expects string input. If non-string data is provided,
        it will log a warning and return the data unchanged, as PII redaction
        is primarily applicable to textual content.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. Not directly used
                                       by this node's redaction logic.

        Returns:
            Any: The data with PII redacted if the input was a string,
                 otherwise the original data.

        Raises:
            RuntimeError: If an unexpected error occurs during the redaction process.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "PII redaction is applicable only to strings. Returning data unchanged."
            )
            return data

        redacted_content = str(data) # Work with a mutable copy of the string

        try:
            for pii_type, pattern in self._pii_patterns.items():
                matches = set(pattern.findall(redacted_content))
                if matches:
                    # Log identified PII for debugging or audit purposes (without actual values)
                    # For a truly secure system, logging actual PII even for debugging would be avoided.
                    logger.debug(
                        f"[{self.node_name}] Found potential '{pii_type}' PII patterns. "
                        f"Example match: '{list(matches)[0][:10]}...' (truncated for privacy)"
                    )
                    redacted_content = pattern.sub(self._redaction_placeholder, redacted_content)
            
            logger.info(f"[{self.node_name}] PII redaction completed.")
            return redacted_content
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during PII redaction: {e}",
                exc_info=True # Log traceback for detailed error analysis
            )
            # Re-raise a more descriptive runtime error to signal a critical failure
            raise RuntimeError(f"[{self.node_name}] Failed to redact PII: {e}") from e