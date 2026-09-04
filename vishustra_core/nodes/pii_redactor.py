import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact Personally Identifiable Information (PII)
    from string data.

    This node employs predefined regular expressions to identify and replace common PII
    patterns, such as email addresses and phone numbers, with generic placeholders.
    This helps in anonymizing sensitive information before it proceeds to subsequent
    processing stages or storage. The node gracefully handles non-string inputs
    by passing them through without modification.
    """

    # Pre-compiled regex patterns for common PII types and their replacements.
    # This dictionary can be extended to include more comprehensive PII categories
    # like social security numbers, credit card numbers, etc., as required.
    _PII_PATTERNS = {
        "email": (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), "[EMAIL_REDACTED]"),
        "phone_us": (re.compile(r'\b(?:\+?1[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b'), "[PHONE_REDACTED]"),
        # Example for a hypothetical SSN pattern (for illustrative purposes; real-world
        # SSN detection requires more robust logic and context):
        # "ssn_us": (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN_REDACTED]"),
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII if the data is a string.

        Args:
            data: The input data, which is expected to be a string potentially containing PII.
                  If the input is not a string, it will be returned unchanged.
            context: A dictionary providing contextual information for the processing.
                     Currently, this basic implementation does not utilize context for
                     dynamic pattern configuration, but it serves as an extension point.

        Returns:
            The processed data with identified PII redacted, or the original data
            if it was not a string or if no PII was found/redacted.
        """
        if not isinstance(data, str):
            logger.debug(
                "Input data for PII Redactor is of type '%s', not a string. Skipping redaction.",
                type(data).__name__
            )
            return data

        redacted_data = data
        for pii_type, (pattern, replacement) in self._PII_PATTERNS.items():
            try:
                new_redacted_data = pattern.sub(replacement, redacted_data)
                if new_redacted_data != redacted_data:
                    logger.debug("Successfully redacted patterns of type '%s'.", pii_type)
                redacted_data = new_redacted_data
            except re.error as e:
                logger.error(
                    "Regex error encountered while redacting PII of type '%s': %s. "
                    "This pattern might be malformed or incompatible.", pii_type, e
                )
            except Exception as e:
                logger.error(
                    "An unexpected error occurred during redaction for PII type '%s': %s. "
                    "Processing continues, but data might be partially unredacted.", pii_type, e
                )

        logger.info("PII redaction process completed.")
        return redacted_data