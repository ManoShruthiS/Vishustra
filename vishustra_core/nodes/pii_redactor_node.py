import logging
import re
from typing import Any, Dict, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node that redacts personally identifiable information (PII)
    from string data using predefined regular expressions.

    It identifies common PII patterns like email addresses, phone numbers,
    and U.S. Social Security Numbers, replacing them with a configurable
    placeholder string.
    """

    # Pre-compiled common PII patterns and their descriptions.
    # These are illustrative and can be expanded or refined based on
    # specific project PII definitions and regulatory requirements.
    _PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Email addresses
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "email address"),
        # U.S. Phone numbers (various common formats)
        (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "U.S. phone number"),
        # U.S. Social Security Numbers (various common formats)
        (re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'), "U.S. Social Security Number"),
        # IP Addresses (IPv4)
        (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'), "IP Address"),
        # Note: More complex PII (e.g., names, addresses) often require context-aware
        # NLP techniques for accurate detection to avoid over-redaction.
    ]

    DEFAULT_REDACTION_PLACEHOLDER = "[REDACTED]"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        If the input `data` is a string, this method applies a set of regular
        expressions to find and replace known PII patterns with a specified
        placeholder. Non-string data is returned unchanged with a logged warning.

        Args:
            data: The input data to be processed. Expected to be a string
                  for PII redaction.
            context: A dictionary containing contextual information.
                     This can include 'redaction_placeholder' (str) to override
                     the default placeholder string used for redaction.

        Returns:
            The data with PII redacted if `data` was a string,
            otherwise the original `data` unmodified.

        Raises:
            Exception: Catches and logs any unexpected errors that occur during
                       the regex processing, returning the original data to
                       maintain pipeline robustness.
        """
        logger.debug(f"[{self.node_name}] Starting PII redaction process.")

        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Input data is not a string (type: {type(data).__name__}). "
                "PII redaction is designed for string data and will be skipped."
            )
            return data

        redaction_placeholder = context.get(
            'redaction_placeholder', self.DEFAULT_REDACTION_PLACEHOLDER
        )
        redacted_data = data
        redaction_counts = {}

        try:
            for pattern, description in self._PII_PATTERNS:
                # Find all matches for the current pattern in the data
                # (which might already be partially redacted by previous patterns)
                matches_found = pattern.findall(redacted_data)

                if matches_found:
                    count = len(matches_found)
                    redacted_data = pattern.sub(redaction_placeholder, redacted_data)
                    redaction_counts[description] = redaction_counts.get(description, 0) + count
                    logger.debug(
                        f"[{self.node_name}] Redacted {count} instance(s) of '{description}'."
                    )

            if redaction_counts:
                summary = ", ".join([f"{count} {desc}" for desc, count in redaaction_counts.items()])
                logger.info(
                    f"[{self.node_name}] Successfully redacted PII. Redaction summary: {summary}"
                )
            else:
                logger.debug(f"[{self.node_name}] No PII patterns were identified or redacted.")

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during PII redaction: {e}",
                exc_info=True
            )
            # In case of an error, return the original data to prevent pipeline failure.
            return data

        logger.debug(f"[{self.node_name}] Finished PII redaction process.")
        return redacted_data
