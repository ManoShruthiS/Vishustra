import logging
import re
from typing import Any, Dict, List, Tuple

# Assuming this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node designed to redact common Personally Identifiable Information (PII)
    from text data.

    This node identifies and replaces sensitive patterns such as email addresses and
    phone numbers with a configurable placeholder (e.g., '[REDACTED_EMAIL]').
    It supports customization of redaction patterns during initialization.
    """

    def __init__(self, patterns: List[Tuple[str, str]] = None):
        """
        Initializes the PIIRedactorNode with optional custom redaction patterns.

        Args:
            patterns: An optional list of tuples, where each tuple contains
                      (regex_pattern_string, replacement_string).
                      If None, a set of default patterns for emails and common phone numbers
                      will be used. All regex patterns will be compiled for performance.
        """
        if patterns:
            self._redaction_patterns = [
                (re.compile(pattern), replacement)
                for pattern, replacement in patterns
            ]
            logger.info("PIIRedactorNode initialized with custom redaction patterns.")
        else:
            # Default PII patterns: email and a basic North American phone number format
            self._redaction_patterns = [
                (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
                (re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[REDACTED_PHONE]'),
                # Add more comprehensive patterns for other PII types (e.g., SSN, credit cards) here
                # if needed for broader PII coverage.
            ]
            logger.info("PIIRedactorNode initialized with default PII redaction patterns.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "PIIRedactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII based on configured patterns.

        If the input `data` is a string, it iterates through all compiled regex patterns
        and replaces any matches with their corresponding replacement strings.
        If `data` is not a string, a warning is logged, and the data is returned unchanged,
        as PII redaction is primarily text-based.

        Args:
            data: The input data, typically expected to be a string that may contain PII.
            context: A dictionary containing contextual information relevant to the
                     orchestration. This node does not modify or directly use the context
                     for its core redaction logic.

        Returns:
            The processed data with PII redacted if it was a string,
            otherwise the original data unchanged.
        """
        logger.debug(f"PIIRedactorNode received data for processing. Type: {type(data)}")

        if not isinstance(data, str):
            logger.warning(
                f"PIIRedactorNode received non-string data (type: {type(data).__name__}). "
                "PII redaction is currently only applicable to string inputs. "
                "Returning data unchanged."
            )
            return data

        redacted_data = data

        for pattern, replacement in self._redaction_patterns:
            try:
                # Apply the redaction pattern
                newly_redacted_data = pattern.sub(replacement, redacted_data)
                if newly_redacted_data != redacted_data:
                    logger.debug(
                        f"Redacted pattern '{pattern.pattern}' with '{replacement}' "
                        f"in data fragment: '{redacted_data[:50]}...'"
                    )
                redacted_data = newly_redacted_data
            except re.error as e:
                logger.error(
                    f"A regular expression error occurred with pattern '{pattern.pattern}': {e}. "
                    "Skipping this pattern for the current data."
                )
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during PII redaction with pattern '{pattern.pattern}': {e}",
                    exc_info=True
                )
                # In case of an unexpected error, it's safer to proceed with partially redacted data
                # rather than failing the entire pipeline.

        if redacted_data != data:
            logger.info("PII redaction successfully applied to the input data.")
        else:
            logger.debug("No identifiable PII patterns were found or redacted in the data.")

        return redacted_data