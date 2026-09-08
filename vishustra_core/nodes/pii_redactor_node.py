import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that identifies and redacts common Personally
    Identifiable Information (PII) patterns from text data.

    This node provides a configurable way to mask sensitive data, ensuring
    privacy compliance for information flowing through the Vishustra framework.
    It currently supports redaction of common patterns like email addresses
    and phone numbers, and can be extended for other PII types.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact common PII patterns.

        The `data` is expected to be a string. If the input is not a string,
        a TypeError will be raised. The redaction logic is applied sequentially
        for defined PII types.

        Args:
            data: The input data, expected to be a string that may contain PII.
            context: A dictionary of contextual information. This can be used
                     in future iterations to pass configuration, such as
                     custom PII patterns or redaction masks.

        Returns:
            The processed data with identified PII redacted, or the original
            data if an unrecoverable error occurs during redaction.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "PIIRedactorNode received non-string input of type '%s'. "
                "PII redaction can only be performed on strings.",
                type(data).__name__
            )
            raise TypeError("PIIRedactorNode can only process string data.")

        redacted_data = data
        redaction_summary = {}

        # Define PII patterns and their redaction replacements.
        # These patterns are illustrative and cover common cases. For production-grade
        # PII detection, consider integrating with specialized NLP libraries.
        pii_configurations = {
            "email": {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "replacement": "[REDACTED_EMAIL]"
            },
            "phone_number_us": {
                "pattern": r'\b(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                "replacement": "[REDACTED_PHONE_NUMBER]"
            },
            "ip_address": {
                "pattern": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
                "replacement": "[REDACTED_IP_ADDRESS]"
            },
            # Example for a more complex PII type, like credit card numbers.
            # Real-world patterns would need to be very robust and potentially
            # validated with checksums for accuracy.
            # "credit_card_number": {
            #     "pattern": r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){3}\d{3,4}\b',
            #     "replacement": "[REDACTED_CREDIT_CARD]"
            # }
        }

        try:
            for pii_type, config in pii_configurations.items():
                pattern = config["pattern"]
                replacement = config["replacement"]
                
                # Apply the redaction using regex substitution
                redacted_data, num_substitutions = re.subn(
                    pattern,
                    replacement,
                    redacted_data
                )

                if num_substitutions > 0:
                    redaction_summary[pii_type] = num_substitutions
                    logger.debug(
                        "Redacted %d instances of '%s' in the data.",
                        num_substitutions,
                        pii_type
                    )

            if redaction_summary:
                logger.info(
                    "Successfully redacted PII types: %s",
                    ", ".join(f"{k} ({v})" for k, v in redaction_summary.items())
                )
            else:
                logger.debug("No PII matching defined patterns was found for redaction.")

            return redacted_data

        except re.error as e:
            logger.error(
                "A regex pattern error occurred during PII redaction: %s. "
                "Returning original data.",
                e
            )
            return data
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during PII redaction. "
                "Returning original data."
            )
            return data # Ensure data is returned even on unexpected errors
