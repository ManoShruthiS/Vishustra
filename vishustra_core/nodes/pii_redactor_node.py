import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personal Identifiable Information (PII)
    from string data. This node identifies common PII patterns like email addresses
    and phone numbers and replaces them with a `[REDACTED_PII]` placeholder.

    The current implementation uses predefined regex patterns. Future enhancements
    could allow for configurable patterns via the `context` or constructor.
    """

    _PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        # Note: More complex PII (e.g., names, addresses without clear delimiters)
        # often requires NLP models or specific entity recognition, which is
        # beyond the scope of this regex-based node.
    }
    _REDACTION_PLACEHOLDER = "[REDACTED_PII]"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        If the input `data` is a string, it will be scanned for predefined PII
        patterns. Identified PII will be replaced with `[REDACTED_PII]`.
        If the input `data` is not a string, it is returned unchanged, and a
        warning is logged.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing potentially sensitive information.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing operation. Not directly
                                       used for configuration in this version,
                                       but available for future extensions.

        Returns:
            Any: The processed data, with PII redacted if it was a string.
                 Returns the original data if it was not a string.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "PII redaction only applies to strings. Returning data unchanged."
            )
            return data

        redacted_data = str(data) # Start with a mutable copy of the string
        found_pii_count = 0

        for pii_type, pattern in self._PII_PATTERNS.items():
            try:
                # Find all occurrences of the PII pattern
                matches = re.findall(pattern, redacted_data, re.IGNORECASE)
                if matches:
                    logger.debug(
                        f"[{self.node_name}] Found {len(matches)} potential '{pii_type}' PII instances. Redacting..."
                    )
                    # Replace all occurrences with the placeholder
                    redacted_data = re.sub(pattern, self._REDACTION_PLACEHOLDER, redacted_data, flags=re.IGNORECASE)
                    found_pii_count += len(matches)
            except re.error as e:
                logger.error(
                    f"[{self.node_name}] Regex error encountered for pattern '{pii_type}': {e}. "
                    "Skipping this pattern."
                )
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] An unexpected error occurred during PII redaction for pattern '{pii_type}': {e}. "
                    "Returning potentially unredacted data."
                )
                return data # Return original data on critical failure

        if found_pii_count > 0:
            logger.info(
                f"[{self.node_name}] Successfully redacted {found_pii_count} PII instances."
            )
        else:
            logger.debug(f"[{self.node_name}] No PII found in the input data.")

        return redacted_data