import logging
import re
from typing import Any, Dict, List, Union

# Simulate the BaseNode import path as per instructions.
# In a real Vishustra project, this would simply be:
# from vishustra_core.nodes.base_node import BaseNode
# For the purpose of this standalone file, the BaseNode definition
# from the project context is included directly.
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """
    Base class for all Vishustra processing nodes.
    Each node must implement the process method.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data and returns the result.
        """
        pass
        
    @property
    @abstractmethod
    def node_name(self) -> str:
        """Returns the name of the node."""
        pass


logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to identify and redact personally
    identifiable information (PII) from text data.

    This node uses predefined regular expressions to detect common PII patterns
    such as email addresses, phone numbers, and credit card numbers, replacing
    them with generic, configurable placeholders.

    The node supports processing of string, dictionary (top-level string values),
    and list (top-level string elements) data types.
    """

    # Class-level dictionary of PII patterns and their respective replacement strings.
    # This design allows for easy extension and potential external configuration
    # in future iterations (e.g., via the node's constructor or context dictionary).
    _PII_PATTERNS = {
        "email": {
            "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "replacement": "[REDACTED_EMAIL]"
        },
        "phone_number": {
            # Matches various international and US phone number formats.
            "regex": r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "replacement": "[REDACTED_PHONE]"
        },
        "credit_card": {
            # A more specific regex attempting to match common CC prefixes.
            # Note: Robust credit card detection typically requires more than regex,
            # such as Luhn algorithm validation, but this serves as a good illustrative example.
            "regex": r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){2}\d{3,4}\b",
            "replacement": "[REDACTED_CREDIT_CARD]"
        },
        # Additional PII types (e.g., social security numbers, physical addresses)
        # can be added here with their corresponding regex patterns and replacements.
    }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "PII Redactor"

    def _redact_text(self, text: str) -> str:
        """
        Applies all configured PII redaction patterns to the given text string.

        Args:
            text: The input string potentially containing PII.

        Returns:
            The string with all identified PII replaced by their respective placeholders.
        """
        original_text = text
        for pii_type, config in self._PII_PATTERNS.items():
            pattern = config["regex"]
            replacement = config["replacement"]
            # re.sub replaces all occurrences of the pattern in the text.
            text, num_subs = re.subn(pattern, replacement, text)
            if num_subs > 0:
                logger.debug(f"Redacted {num_subs} instances of {pii_type} in text.")
        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII based on predefined patterns.

        This method supports multiple input data types:
        - If 'data' is a `str`, PII redaction patterns are applied directly.
        - If 'data' is a `dict`, it iterates through top-level string values and redacts them.
          Non-string values or nested structures within the dictionary are left untouched.
        - If 'data' is a `list`, it iterates through top-level string elements and redacts them.
          Non-string elements or nested structures within the list are left untouched.
        - For any other data type, a warning is logged, and the data is returned without modification.

        Args:
            data: The input data, which can be a `str`, `dict`, `list`, or any other type.
            context: A dictionary containing contextual information for the processing.
                     While not directly used for PII patterns in this version, it's
                     available for future enhancements, such as passing custom patterns
                     or redaction masks dynamically.

        Returns:
            The processed data with identified PII redacted. If the input data type
            is not supported for redaction, the original data is returned as-is.
        """
        if isinstance(data, str):
            logger.debug("PIIRedactorNode: Processing string data.")
            return self._redact_text(data)
        elif isinstance(data, dict):
            logger.debug("PIIRedactorNode: Processing dictionary data.")
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    processed_data[key] = self._redact_text(value)
                else:
                    # Preserve non-string values or nested structures as-is
                    processed_data[key] = value
            return processed_data
        elif isinstance(data, list):
            logger.debug("PIIRedactorNode: Processing list data.")
            processed_data = []
            for item in data:
                if isinstance(item, str):
                    processed_data.append(self._redact_text(item))
                else:
                    # Preserve non-string items or nested structures as-is
                    processed_data.append(item)
            return processed_data
        else:
            logger.warning(
                f"PIIRedactorNode received unsupported data type: {type(data).__name__}. "
                "No PII redaction performed; data returned as-is."
            )
            return data