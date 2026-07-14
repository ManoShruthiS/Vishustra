import re
import logging
from typing import Any, Dict, List, Tuple
from vishustra_core.nodes.base_node import BaseNode

# Set up logging for the module
logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts personally identifiable information (PII)
    from string data using predefined regular expressions.
    """

    _PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
        (re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE_REDACTED]'),
        (re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'), '[SSN_REDACTED]'),
        (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{3,4}\b'), '[CREDIT_CARD_REDACTED]'),
        (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b'), '[PASSPORT_REDACTED]'), # Example for a generic passport format
        (re.compile(r'\b(?:[A-Z][a-z]+(?: [A-Z][a-z]+)+)\b'), '[NAME_REDACTED]') # Very simplistic name detection, often requires NLP
    ]

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "pii_redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Redacts PII from the input data.

        Expected input `data` is a string. If `data` is not a string,
        a TypeError is raised. The `context` dictionary is currently
        not used by this node but is provided by the BaseNode interface.

        Args:
            data: The input string data to be processed for PII redaction.
            context: A dictionary containing contextual information
                     for the processing pipeline.

        Returns:
            The input data string with identified PII redacted.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"PIIRedactorNode received non-string data: {type(data)}. Expected string.")
            raise TypeError(f"PIIRedactorNode expects string data, but received {type(data)}.")

        redacted_data = data
        redactions_made = 0

        for pattern, replacement_tag in self._PII_PATTERNS:
            original_data = redacted_data
            redacted_data = pattern.sub(replacement_tag, redacted_data)
            if original_data != redacted_data:
                count = len(pattern.findall(original_data)) # Count occurrences *before* sub
                logger.debug(f"Redacted {count} occurrences of {replacement_tag} using pattern: {pattern.pattern}")
                redactions_made += count

        if redactions_made > 0:
            logger.info(f"Successfully redacted {redactions_made} PII entities from the data.")
        else:
            logger.debug("No PII entities found for redaction in the data.")

        return redacted_data