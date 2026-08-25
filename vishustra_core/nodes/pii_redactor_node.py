import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per Vishustra's module structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node designed to redact Personally Identifiable Information (PII)
    from text data. It leverages regular expressions to identify common PII patterns
    such as email addresses, phone numbers, social security numbers, and credit card
    numbers. Matched PII is replaced with generic '[REDACTED_{TYPE}]' placeholders
    to ensure data privacy.

    This node provides recursive processing capabilities, handling strings,
    dictionaries, and lists to redact PII nested within complex data structures.
    Other data types are passed through unchanged, with a logged warning.
    """

    # Define PII patterns using regular expressions. These patterns are illustrative
    # and can be extended or configured based on specific project requirements and
    # compliance standards.
    _PII_PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE_NUMBER": r'\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b',
        "SSN": r'\b\d{3}[- ]\d{2}[- ]\d{4}\b', # Simplified U.S. SSN format
        "CREDIT_CARD": r'\b(?:\d{4}[ -]?){3}\d{4}\b' # Simplified 16-digit credit card format
    }

    def __init__(self):
        super().__init__()
        # Pre-compile all regex patterns during initialization for improved performance
        # when processing multiple data batches or complex data structures.
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self._PII_PATTERNS.items()
        }
        logger.debug(
            f"{self.node_name} initialized with PII detection patterns: "
            f"{list(self._compiled_patterns.keys())}"
        )

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this PII Redactor node.
        """
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies PII redaction to a single string. It iterates through all defined
        PII patterns, replacing any identified matches with a type-specific
        redacted placeholder. Detailed logging is performed for each redaction.

        Args:
            text: The input string potentially containing PII.

        Returns:
            The string with all detected PII redacted.
        """
        redacted_text = text
        for pii_type, pattern_re in self._compiled_patterns.items():
            # Find all matches first to enable logging of each instance before replacement.
            matches = pattern_re.findall(redacted_text)
            if matches:
                logger.debug(f"Found {len(matches)} instance(s) of {pii_type} for redaction.")
                # Log each matched PII segment for detailed auditing or debugging.
                for match in matches:
                    logger.debug(f"Identified {pii_type} for redaction: '{match}'")

                # Replace all occurrences of the current PII type with its placeholder.
                redacted_text = pattern_re.sub(f"[REDACTED_{pii_type}]", redacted_text)
                logger.info(f"Redacted {len(matches)} instance(s) of {pii_type}.")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, recursively redacting PII from string values
        found within the data. This method handles various data types:
        strings, dictionaries, and lists.

        Args:
            data: The input data, which can be a string, dict, list, or other types.
            context: A dictionary providing runtime context. This parameter is
                     available for future extensions, such as dynamic pattern
                     configuration, but is not directly utilized in this basic
                     implementation for PII redaction logic.

        Returns:
            The transformed data with PII redacted. If the data type is not directly
            supported for content-based redaction (e.g., integers, booleans, floats),
            the data is returned unchanged after a warning is logged, ensuring
            non-disruptive processing for unsupported types.
        """
        logger.debug(f"{self.node_name} received data for processing. Input type: {type(data).__name__}")

        if isinstance(data, str):
            # If the data itself is a string, redact it directly.
            return self._redact_string(data)
        elif isinstance(data, dict):
            # If data is a dictionary, recursively process each value.
            redacted_dict = {}
            for key, value in data.items():
                redacted_dict[key] = self.process(value, context)
            return redacted_dict
        elif isinstance(data, list):
            # If data is a list, recursively process each item.
            redacted_list = []
            for item in data:
                redacted_list.append(self.process(item, context))
            return redacted_list
        else:
            # For data types not explicitly handled for PII redaction,
            # log a warning and return the data unchanged.
            logger.warning(
                f"{self.node_name}: Data type '{type(data).__name__}' is not directly "
                "processed for PII redaction. Returning data unchanged."
            )
            return data